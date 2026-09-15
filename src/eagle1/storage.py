"""Persistance des episodes joues, en SQLite.

Le brief impose un prototype 100 % local, sans serveur externe : SQLite est
dans la bibliotheque standard, ne demande aucun service a demarrer et suffit
largement au volume attendu.

Deux tables : le resume de chaque episode, et le detail pas a pas qui
alimente l'analyse des decisions par circonstance.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .config import DATA_DIR

CHEMIN_BASE = DATA_DIR / "episodes.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id                TEXT PRIMARY KEY,
    cree_le           TEXT NOT NULL,
    modele            TEXT NOT NULL,
    seed              INTEGER,
    recompense_totale REAL NOT NULL,
    nb_pas            INTEGER NOT NULL,
    reussi            INTEGER NOT NULL,
    pose              INTEGER NOT NULL,
    video             TEXT
);

CREATE TABLE IF NOT EXISTS pas (
    episode_id   TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    t            INTEGER NOT NULL,
    observation  TEXT NOT NULL,
    action       INTEGER NOT NULL,
    action_nom   TEXT NOT NULL,
    recompense   REAL NOT NULL,
    altitude     TEXT NOT NULL,
    inclinaison  TEXT NOT NULL,
    descente     TEXT NOT NULL,
    position     TEXT NOT NULL,
    PRIMARY KEY (episode_id, t)
);

-- Le tableau de bord filtre en permanence sur les circonstances.
CREATE INDEX IF NOT EXISTS idx_pas_circonstances ON pas (inclinaison, altitude, descente);
"""


def connexion() -> sqlite3.Connection:
    """Ouvre la base et garantit que le schema existe."""
    conn = sqlite3.connect(CHEMIN_BASE)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def enregistrer(episode: dict, modele: str) -> str:
    """Stocke un episode et tous ses pas en une seule transaction."""
    with connexion() as conn:
        conn.execute(
            "INSERT INTO episodes (id, cree_le, modele, seed, recompense_totale,"
            " nb_pas, reussi, pose, video) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                episode["episode_id"],
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                modele,
                episode["seed"],
                episode["recompense_totale"],
                episode["nb_pas"],
                int(episode["reussi"]),
                int(episode["pose"]),
                episode["video"],
            ),
        )
        conn.executemany(
            "INSERT INTO pas (episode_id, t, observation, action, action_nom,"
            " recompense, altitude, inclinaison, descente, position)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    episode["episode_id"],
                    p["t"],
                    json.dumps(p["observation"]),
                    p["action"],
                    p["action_nom"],
                    p["recompense"],
                    p["altitude"],
                    p["inclinaison"],
                    p["descente"],
                    p["position"],
                )
                for p in episode["pas"]
            ],
        )
    return episode["episode_id"]


def lister(limite: int = 50, decalage: int = 0) -> list[dict]:
    with connexion() as conn:
        lignes = conn.execute(
            "SELECT * FROM episodes ORDER BY cree_le DESC LIMIT ? OFFSET ?",
            (limite, decalage),
        ).fetchall()
    return [dict(ligne) for ligne in lignes]


def compter() -> int:
    with connexion() as conn:
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]


def charger(episode_id: str) -> dict | None:
    """Retourne un episode avec sa trajectoire complete."""
    with connexion() as conn:
        entete = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if entete is None:
            return None
        pas = conn.execute(
            "SELECT * FROM pas WHERE episode_id = ? ORDER BY t", (episode_id,)
        ).fetchall()

    episode = dict(entete)
    episode["pas"] = [{**dict(p), "observation": json.loads(p["observation"])} for p in pas]
    return episode


def metriques() -> dict:
    """Agregats sur tous les episodes joues, pour le tableau de bord.

    Les statistiques sont calculees en SQL plutot qu'en Python : inutile de
    rapatrier des milliers de lignes pour en faire une moyenne.
    """
    with connexion() as conn:
        base = conn.execute(
            "SELECT COUNT(*) n, AVG(recompense_totale) moyenne,"
            " MIN(recompense_totale) minimum, MAX(recompense_totale) maximum,"
            " AVG(nb_pas) duree_moyenne, AVG(reussi) taux_reussite"
            " FROM episodes"
        ).fetchone()

        recompenses = [r[0] for r in conn.execute("SELECT recompense_totale FROM episodes")]

        actions = conn.execute(
            "SELECT action_nom, COUNT(*) n FROM pas GROUP BY action_nom ORDER BY n DESC"
        ).fetchall()

        # Croisement circonstance x action : l'exigence explicite de l'enonce.
        decisions = {}
        for famille in ("altitude", "inclinaison", "descente", "position"):
            lignes = conn.execute(
                f"SELECT {famille} situation, action_nom, COUNT(*) n"
                f" FROM pas GROUP BY {famille}, action_nom"
            ).fetchall()
            famille_resume: dict[str, dict[str, int]] = {}
            for ligne in lignes:
                famille_resume.setdefault(ligne["situation"], {})[ligne["action_nom"]] = ligne["n"]
            decisions[famille] = famille_resume

    n = base["n"] or 0
    if n:
        moyenne = base["moyenne"]
        ecart_type = (sum((r - moyenne) ** 2 for r in recompenses) / n) ** 0.5
    else:
        ecart_type = 0.0

    return {
        "nb_episodes": n,
        "recompense_moyenne": round(base["moyenne"], 2) if n else None,
        "ecart_type": round(ecart_type, 2),
        "recompense_min": base["minimum"],
        "recompense_max": base["maximum"],
        "duree_moyenne": round(base["duree_moyenne"], 1) if n else None,
        "taux_reussite": round(base["taux_reussite"], 4) if n else None,
        "recompenses": recompenses,
        "repartition_actions": {ligne["action_nom"]: ligne["n"] for ligne in actions},
        "decisions_par_circonstance": decisions,
    }
