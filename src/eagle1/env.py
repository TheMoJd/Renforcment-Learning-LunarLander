"""Construction de l'environnement LunarLander-v3.

Point d'entree unique : toute creation d'environnement dans le projet passe
par make_env(). Cela garantit que l'entrainement, l'evaluation, l'API et la
video voient exactement le meme environnement, avec les memes wrappers.
"""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from .config import ENV_ID


def make_env(
    render_mode: str | None = None,
    seed: int | None = None,
    monitor: bool = True,
):
    """Cree un environnement LunarLander-v3 pret a l'emploi.

    Args:
        render_mode: None pour l'entrainement (rapide), "rgb_array" pour
            recuperer des images (GUI, video), "human" pour une fenetre.
        seed: graine appliquee a l'environnement et a l'espace d'actions.
        monitor: enveloppe avec Monitor, qui enregistre la recompense et la
            longueur de chaque episode. C'est cette couche qui alimente
            `rollout/ep_rew_mean` dans TensorBoard -- sans elle, les courbes
            d'entrainement sont vides.
    """
    env = gym.make(ENV_ID, render_mode=render_mode)

    if monitor:
        env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def describe_env() -> dict:
    """Retourne une description des espaces, pour la section exploration
    du notebook et pour l'endpoint /model de l'API."""
    env = gym.make(ENV_ID)
    description = {
        "env_id": ENV_ID,
        "observation_space": str(env.observation_space),
        "observation_dim": int(env.observation_space.shape[0]),
        "action_space": str(env.action_space),
        "n_actions": int(env.action_space.n),
        "action_meanings": [
            "ne rien faire",
            "propulseur gauche",
            "propulseur principal",
            "propulseur droit",
        ],
        "observation_meanings": [
            "position x",
            "position y",
            "vitesse x",
            "vitesse y",
            "angle",
            "vitesse angulaire",
            "contact pied gauche",
            "contact pied droit",
        ],
    }
    env.close()
    return description
