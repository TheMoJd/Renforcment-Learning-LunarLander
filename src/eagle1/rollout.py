"""Execution d'un episode complet, pas a pas.

L'enonce demande que le tableau de bord montre "les decisions prises par
l'IA en fonction du type de circonstance". Il ne suffit donc pas de stocker
le resultat d'un episode : il faut conserver, a chaque pas, l'etat observe
et l'action choisie, puis traduire l'etat en circonstances lisibles par un
humain.

C'est ici que vit toute la logique RL cote serveur. Le GUI ne fait que lire
ce que ce module produit, via l'API.
"""

from __future__ import annotations

import time
import uuid

import numpy as np

from .config import ENV_ID, SUCCESS_THRESHOLD, VIDEOS_DIR
from .env import make_env

# Signification des 4 actions discretes de LunarLander-v3.
NOMS_ACTIONS = ["rien", "propulseur gauche", "propulseur principal", "propulseur droit"]

# Signification des 8 variables d'observation.
NOMS_OBSERVATIONS = [
    "x", "y", "vx", "vy", "angle", "vitesse_angulaire", "contact_gauche", "contact_droit",
]


def circonstance(obs: np.ndarray) -> dict[str, str]:
    """Traduit une observation numerique en situation comprehensible.

    Les seuils sont volontairement grossiers : l'objectif est de produire des
    categories qu'un humain peut interpreter sur un tableau de bord, pas une
    segmentation fine.

    Convention d'angle verifiee experimentalement : le propulseur gauche
    (action 1) fait augmenter l'angle et deplace l'appareil vers la gauche.
    Un angle positif correspond donc a une inclinaison vers la gauche, que
    l'agent doit corriger avec le propulseur droit.
    """
    x, y, vx, vy, angle, _, contact_g, contact_d = obs

    if contact_g or contact_d:
        altitude = "au sol"
    elif y > 0.8:
        altitude = "haute"
    elif y > 0.3:
        altitude = "moyenne"
    else:
        altitude = "basse"

    if angle > 0.1:
        inclinaison = "penche a gauche"
    elif angle < -0.1:
        inclinaison = "penche a droite"
    else:
        inclinaison = "droit"

    if vy < -0.5:
        descente = "rapide"
    elif vy < -0.1:
        descente = "moderee"
    else:
        descente = "lente ou montee"

    if x < -0.1:
        position = "a gauche de la cible"
    elif x > 0.1:
        position = "a droite de la cible"
    else:
        position = "au-dessus de la cible"

    return {
        "altitude": altitude,
        "inclinaison": inclinaison,
        "descente": descente,
        "position": position,
    }


def jouer_episode(
    model,
    seed: int | None = None,
    enregistrer_video: bool = False,
    max_steps: int = 1000,
) -> dict:
    """Joue un episode complet et retourne sa trajectoire detaillee.

    Args:
        model: agent SB3 deja charge.
        seed: graine de l'episode. Une meme graine rejoue exactement le meme
            terrain et la meme trajectoire -- indispensable pour que le GUI
            puisse reafficher un episode sans le restocker entierement.
        enregistrer_video: produit un .mp4 de l'episode dans videos/.
    """
    episode_id = uuid.uuid4().hex[:12]
    env = make_env(render_mode="rgb_array" if enregistrer_video else None, monitor=False)

    obs, _ = env.reset(seed=seed)
    frames, pas = [], []
    recompense_totale = 0.0
    debut = time.time()

    for t in range(max_steps):
        if enregistrer_video:
            frames.append(env.render())

        # deterministic=True : on montre la politique apprise, sans le bruit
        # d'exploration. Voir la note d'evaluation dans le README.
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        pas.append(
            {
                "t": t,
                "observation": [round(float(v), 4) for v in obs],
                "action": action,
                "action_nom": NOMS_ACTIONS[action],
                **circonstance(obs),
            }
        )

        obs, recompense, termine, tronque, _ = env.step(action)
        pas[-1]["recompense"] = round(float(recompense), 3)
        recompense_totale += float(recompense)

        if termine or tronque:
            break

    env.close()

    chemin_video = None
    if enregistrer_video and frames:
        chemin_video = _ecrire_video(frames, episode_id)

    return {
        "episode_id": episode_id,
        "env_id": ENV_ID,
        "seed": seed,
        "recompense_totale": round(recompense_totale, 2),
        "nb_pas": len(pas),
        "reussi": recompense_totale >= SUCCESS_THRESHOLD,
        # Un atterrissage est "pose" si au moins un pied touche a la fin.
        "pose": bool(obs[6] or obs[7]),
        "duree_calcul_s": round(time.time() - debut, 2),
        "video": chemin_video,
        "pas": pas,
    }


def _ecrire_video(frames: list, episode_id: str) -> str:
    """Encode les images en .mp4. Import local : imageio n'est necessaire
    que lorsqu'une video est demandee."""
    import imageio.v2 as imageio

    chemin = VIDEOS_DIR / f"episode_{episode_id}.mp4"
    # LunarLander tourne a 50 images/seconde : c'est le fps a respecter pour
    # que la video s'ecoule a la vitesse reelle de la simulation.
    imageio.mimsave(chemin, frames, fps=50, macro_block_size=1)
    # as_posix() : le chemin est stocke en base et relu potentiellement
    # sur un autre systeme. Un antislash Windows y serait illisible.
    return chemin.relative_to(VIDEOS_DIR.parent).as_posix()


def resumer_decisions(pas: list[dict]) -> dict:
    """Croise circonstances et actions : coeur du tableau de bord.

    Retourne, pour chaque famille de circonstance et chaque situation, la
    repartition des actions choisies par l'agent. Permet de repondre a
    "quand l'appareil penche a droite, que fait l'IA ?".
    """
    familles = ("altitude", "inclinaison", "descente", "position")
    resume: dict[str, dict[str, dict[str, int]]] = {f: {} for f in familles}

    for famille in familles:
        for etape in pas:
            situation = etape[famille]
            compteur = resume[famille].setdefault(situation, dict.fromkeys(NOMS_ACTIONS, 0))
            compteur[etape["action_nom"]] += 1

    return resume
