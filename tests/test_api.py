"""Tests de bout en bout de l'API.

Chaque test rejoue un episode reel avec le modele entraine : c'est lent
(quelques secondes) mais c'est le seul moyen de verifier que la chaine
complete -- chargement du modele, simulation, persistance, agregation --
fonctionne.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eagle1 import storage
from eagle1.rollout import NOMS_ACTIONS


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Client de test utilisant une base SQLite jetable.

    Sans cela, les tests pollueraient data/episodes.db avec des parties
    fictives, qui apparaitraient ensuite dans le tableau de bord.
    """
    storage.CHEMIN_BASE = tmp_path_factory.mktemp("base") / "test.db"
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_le_service_repond(client) -> None:
    reponse = client.get("/health").json()
    assert reponse["statut"] == "ok"
    assert reponse["modele_charge"] is True


def test_le_modele_servi_atteint_le_seuil_du_sujet(client) -> None:
    """Le modele expose doit etre celui qui depasse 200 sur 100 episodes."""
    evaluation = client.get("/model").json()["evaluation_100_episodes"]
    assert evaluation["objectif_atteint"] is True
    assert evaluation["recompense_moyenne"] >= 200


def test_une_meme_seed_rejoue_le_meme_episode(client) -> None:
    """Reproductibilite : le GUI doit pouvoir reafficher une partie."""
    premier = client.post("/episodes", json={"seed": 123}).json()
    second = client.post("/episodes", json={"seed": 123}).json()
    assert premier["recompense_totale"] == second["recompense_totale"]
    assert premier["nb_pas"] == second["nb_pas"]


def test_un_episode_est_persiste_avec_sa_trajectoire(client) -> None:
    resume = client.post("/episodes", json={"seed": 7}).json()
    detail = client.get(f"/episodes/{resume['episode_id']}").json()

    assert len(detail["pas"]) == resume["nb_pas"]
    premier_pas = detail["pas"][0]
    assert len(premier_pas["observation"]) == 8
    assert premier_pas["action_nom"] in NOMS_ACTIONS
    # L'exigence de l'enonce : chaque pas porte sa circonstance.
    for famille in ("altitude", "inclinaison", "descente", "position"):
        assert premier_pas[famille]


def test_les_metriques_croisent_circonstances_et_actions(client) -> None:
    """Exigence explicite : 'les decisions prises par l'IA en fonction du
    type de circonstance'."""
    metriques = client.get("/metrics").json()

    assert metriques["nb_episodes"] > 0
    assert metriques["ecart_type"] >= 0
    assert set(metriques["decisions_par_circonstance"]) == {
        "altitude", "inclinaison", "descente", "position",
    }
    inclinaisons = metriques["decisions_par_circonstance"]["inclinaison"]
    assert inclinaisons, "aucune decision enregistree"
    for repartition in inclinaisons.values():
        assert set(repartition) <= set(NOMS_ACTIONS)


def test_le_comparatif_des_experiences_est_servi_par_l_api(client) -> None:
    """Le tableau de bord doit comparer les experiences sans lire models/ :
    la contrainte du sujet impose que le frontend ne parle que HTTP."""
    donnees = client.get("/experiments").json()

    assert donnees["nb_experiences"] >= 2
    runs = {experience["run"] for experience in donnees["experiences"]}
    assert donnees["modele_servi"] in runs

    # Trie par moyenne decroissante : le tableau de bord s'appuie sur cet ordre.
    moyennes = [experience["mean"] for experience in donnees["experiences"]]
    assert moyennes == sorted(moyennes, reverse=True)

    meilleure = donnees["experiences"][0]
    assert meilleure["objectif_atteint"] is True
    assert meilleure["mean"] >= 200
    # L'erreur-type est ce qui permet de dire si deux experiences different.
    assert meilleure["sem"] > 0
    assert len(meilleure["recompenses"]) == meilleure["n_episodes"]


def test_episode_inexistant_renvoie_404(client) -> None:
    assert client.get("/episodes/inexistant").status_code == 404
