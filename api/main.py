"""API du pilote automatique Eagle-1.

Contrainte structurante du sujet : toute la logique RL vit ici. L'API charge
le modele, instancie l'environnement et joue les episodes de bout en bout.
Le GUI et le tableau de bord ne font que consommer ces routes en HTTP ; ils
n'importent jamais gymnasium, torch ni stable-baselines3 (un test le
verifie).

Le sujet suggerait un endpoint /play recevant un etat et renvoyant une
action. Il n'est volontairement pas implemente : un tel contrat obligerait
le frontend a faire tourner lui-meme l'environnement, donc a heberger de la
logique RL. Voir le README pour la justification complete.

Lancement :
    uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from eagle1 import storage
from eagle1.config import BEST_RUN, ENV_ID, MODELS_DIR, SUCCESS_THRESHOLD
from eagle1.env import describe_env
from eagle1.evaluate import load_model
from eagle1.rollout import NOMS_ACTIONS, jouer_episode, resumer_decisions

# Le modele est charge une seule fois au demarrage : le recharger a chaque
# requete couterait ~1 s et rendrait l'API inutilisable pour le GUI.
etat: dict = {}


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    modele, metadonnees = load_model(BEST_RUN)
    etat["modele"] = modele
    etat["metadonnees"] = metadonnees
    yield
    etat.clear()


app = FastAPI(
    title="Eagle-1 — pilote automatique d'alunissage",
    description="Joue des episodes de LunarLander-v3 avec l'agent entraine et expose les metriques.",
    version="1.0.0",
    lifespan=cycle_de_vie,
)


class DemandeEpisode(BaseModel):
    seed: int | None = Field(
        default=None,
        description="Graine de l'episode. Une meme graine rejoue exactement la meme partie.",
    )
    video: bool = Field(
        default=False,
        description="Produit un .mp4 de l'episode. Plus lent : la simulation doit etre rendue image par image.",
    )


@app.get("/health", summary="Etat du service")
def health() -> dict:
    return {
        "statut": "ok",
        "modele_charge": "modele" in etat,
        "env": ENV_ID,
        "episodes_en_base": storage.compter(),
    }


@app.get("/model", summary="Description du modele servi")
def model() -> dict:
    """Algorithme, hyperparametres et performance mesuree de l'agent servi."""
    if "metadonnees" not in etat:
        raise HTTPException(503, "Modele non charge")

    fichier_eval = MODELS_DIR / BEST_RUN / "eval_best_model_100ep.json"
    evaluation = json.loads(fichier_eval.read_text(encoding="utf-8")) if fichier_eval.exists() else None

    return {
        "run": BEST_RUN,
        **etat["metadonnees"],
        "environnement": describe_env(),
        "seuil_reussite": SUCCESS_THRESHOLD,
        "evaluation_100_episodes": (
            {
                "recompense_moyenne": round(evaluation["mean_reward"], 2),
                "ecart_type": round(evaluation["std_reward"], 2),
                "taux_reussite": evaluation["success_rate"],
                "objectif_atteint": evaluation["passes_requirement"],
            }
            if evaluation
            else None
        ),
    }


@app.post("/episodes", status_code=201, summary="Joue un episode")
def jouer(demande: DemandeEpisode) -> dict:
    """Fait atterrir Eagle-1 une fois et enregistre la partie.

    Retourne le resume plus la repartition des actions par circonstance. La
    trajectoire complete reste accessible via GET /episodes/{id}.
    """
    if "modele" not in etat:
        raise HTTPException(503, "Modele non charge")

    episode = jouer_episode(etat["modele"], seed=demande.seed, enregistrer_video=demande.video)
    storage.enregistrer(episode, modele=BEST_RUN)

    return {
        "episode_id": episode["episode_id"],
        "seed": episode["seed"],
        "recompense_totale": episode["recompense_totale"],
        "nb_pas": episode["nb_pas"],
        "reussi": episode["reussi"],
        "pose": episode["pose"],
        "video": episode["video"],
        "decisions_par_circonstance": resumer_decisions(episode["pas"]),
    }


@app.get("/episodes", summary="Liste des episodes joues")
def lister(
    limite: int = Query(default=50, ge=1, le=500),
    decalage: int = Query(default=0, ge=0),
) -> dict:
    return {"total": storage.compter(), "episodes": storage.lister(limite, decalage)}


@app.get("/episodes/{episode_id}", summary="Trajectoire complete d'un episode")
def detail(episode_id: str) -> dict:
    """Etat, action et circonstance a chaque pas de temps."""
    episode = storage.charger(episode_id)
    if episode is None:
        raise HTTPException(404, f"Episode {episode_id} introuvable")
    episode["decisions_par_circonstance"] = resumer_decisions(episode["pas"])
    return episode


@app.get("/episodes/{episode_id}/video", summary="Video d'un episode")
def video(episode_id: str) -> FileResponse:
    """Le .mp4 n'existe que si l'episode a ete joue avec video=true."""
    episode = storage.charger(episode_id)
    if episode is None:
        raise HTTPException(404, f"Episode {episode_id} introuvable")
    if not episode["video"]:
        raise HTTPException(404, "Cet episode a ete joue sans enregistrement video")

    from eagle1.config import ROOT

    chemin = ROOT / episode["video"]
    if not chemin.exists():
        raise HTTPException(404, "Fichier video introuvable sur le disque")
    return FileResponse(chemin, media_type="video/mp4", filename=chemin.name)


@app.get("/experiments", summary="Comparatif des experiences d'entrainement")
def experiments() -> dict:
    """Resultats des douze experiences d'hyperparametres, triees par moyenne.

    Le tableau de bord doit pouvoir montrer cette comparaison, mais il n'a
    pas le droit de lire models/ directement : cette route lui sert les
    memes donnees en HTTP.
    """
    from eagle1.experiments import resultats

    lignes = resultats()
    return {
        "seuil_reussite": SUCCESS_THRESHOLD,
        "modele_servi": BEST_RUN,
        "nb_experiences": len(lignes),
        "experiences": lignes,
    }


@app.get("/metrics", summary="Metriques agregees")
def metrics() -> dict:
    """Alimente le tableau de bord : moyenne, ecart-type, taux de reussite,
    repartition des actions et decisions par circonstance."""
    return {
        "seuil_reussite": SUCCESS_THRESHOLD,
        "actions_possibles": NOMS_ACTIONS,
        **storage.metriques(),
    }
