"""Evaluation d'un agent entraine.

Le sujet impose l'usage de `evaluate_policy` et exige DEUX chiffres :
la recompense moyenne ET l'ecart-type ("le score cible de 200 doit etre une
moyenne stable, pas un coup de chance").

Usage :
    python -m eagle1.evaluate --run dqn_baseline --episodes 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from .config import (
    EVAL_EPISODES_FINAL,
    EVAL_SEED,
    MODELS_DIR,
    SUCCESS_THRESHOLD,
)
from .env import make_env
from .train import ALGOS


def load_model(run: str, checkpoint: str = "best_model"):
    """Charge un modele depuis models/<run>/<checkpoint>.zip.

    L'algorithme est relu dans metadata.json : un .zip SB3 ne peut pas etre
    charge sans savoir s'il s'agit d'un DQN ou d'un PPO.
    """
    run_dir = MODELS_DIR / run
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json absent dans {run_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = run_dir / f"{checkpoint}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")

    return ALGOS[metadata["algo"]].load(model_path), metadata


def evaluate_run(
    run: str,
    n_episodes: int = EVAL_EPISODES_FINAL,
    checkpoint: str = "best_model",
    seed: int = EVAL_SEED,
) -> dict:
    """Evalue un agent et retourne les statistiques du sujet.

    deterministic=True : on mesure la politique apprise, sans le bruit
    d'exploration. Un DQN qui garde epsilon > 0 en evaluation se sabote.
    """
    model, metadata = load_model(run, checkpoint)
    env = make_env(seed=seed)

    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    env.close()

    rewards_array = np.asarray(rewards, dtype=float)
    stats = {
        "run": run,
        "checkpoint": checkpoint,
        "algo": metadata["algo"],
        "preset": metadata.get("preset"),
        "n_episodes": n_episodes,
        "eval_seed": seed,
        "mean_reward": float(rewards_array.mean()),
        "std_reward": float(rewards_array.std()),
        "min_reward": float(rewards_array.min()),
        "max_reward": float(rewards_array.max()),
        "mean_length": float(np.mean(lengths)),
        # Taux de succes : part des episodes qui franchissent seuls le seuil
        # de 200. Complementaire de la moyenne -- une moyenne de 200 avec un
        # gros ecart-type cache souvent des crashes.
        "success_rate": float((rewards_array >= SUCCESS_THRESHOLD).mean()),
        "passes_requirement": bool(rewards_array.mean() >= SUCCESS_THRESHOLD),
        "episode_rewards": rewards_array.round(2).tolist(),
    }

    # Sauvegarde a cote du modele : le dashboard et le notebook relisent ce
    # fichier au lieu de relancer une evaluation de plusieurs minutes.
    output = MODELS_DIR / run / f"eval_{checkpoint}_{n_episodes}ep.json"
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def evaluate_random_policy(
    n_episodes: int = EVAL_EPISODES_FINAL,
    seed: int = EVAL_SEED,
) -> dict:
    """Mesure une politique purement aleatoire, meme protocole que l'agent.

    Sert de plancher de reference dans la section exploration du notebook :
    sans ce chiffre, impossible de dire si un agent a reellement appris
    quelque chose ou s'il beneficie juste de la forme du reward.
    """
    env = make_env(seed=seed)
    rewards, lengths = [], []

    for episode in range(n_episodes):
        env.reset(seed=seed + episode)
        total, steps, done = 0.0, 0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total)
        lengths.append(steps)

    env.close()
    rewards_array = np.asarray(rewards, dtype=float)
    return {
        "run": "random",
        "checkpoint": "-",
        "algo": "random",
        "preset": None,
        "n_episodes": n_episodes,
        "eval_seed": seed,
        "mean_reward": float(rewards_array.mean()),
        "std_reward": float(rewards_array.std()),
        "min_reward": float(rewards_array.min()),
        "max_reward": float(rewards_array.max()),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float((rewards_array >= SUCCESS_THRESHOLD).mean()),
        "passes_requirement": bool(rewards_array.mean() >= SUCCESS_THRESHOLD),
        "episode_rewards": rewards_array.round(2).tolist(),
    }


def format_stats(stats: dict) -> str:
    """Resume lisible, utilise par la CLI et par le notebook."""
    verdict = "ATTEINT" if stats["passes_requirement"] else "NON ATTEINT"
    return (
        f"{stats['run']} / {stats['checkpoint']} ({stats['algo'].upper()})\n"
        f"  Recompense moyenne : {stats['mean_reward']:.1f} "
        f"+/- {stats['std_reward']:.1f}  (sur {stats['n_episodes']} episodes)\n"
        f"  Min / Max          : {stats['min_reward']:.1f} / {stats['max_reward']:.1f}\n"
        f"  Taux de succes     : {stats['success_rate']:.0%} d'episodes >= 200\n"
        f"  Duree moyenne      : {stats['mean_length']:.0f} steps\n"
        f"  Seuil des 200      : {verdict}"
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evalue un agent Eagle-1")
    parser.add_argument("--run", required=True, help="nom du run dans models/")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES_FINAL)
    parser.add_argument("--checkpoint", default="best_model")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    args = parser.parse_args()
    print(format_stats(evaluate_run(args.run, args.episodes, args.checkpoint, args.seed)))


if __name__ == "__main__":
    _cli()
