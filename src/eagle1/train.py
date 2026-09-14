"""Entrainement d'un agent RL sur LunarLander-v3.

Usage :
    python -m eagle1.train --algo dqn --preset baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback

from .config import (
    DEFAULT_TIMESTEPS,
    EVAL_SEED,
    MODELS_DIR,
    PRESETS,
    RUNS_DIR,
    TRAIN_SEED,
)
from .env import make_env

# LunarLander-v3 a un espace d'actions DISCRET (4 actions), donc DQN est
# l'algorithme attendu. PPO est conserve comme comparatif : contrairement a
# ce qu'affirme le brief PDF, PPO n'est pas reserve au continu, il gere
# tres bien le discret -- c'est justement l'interet de la comparaison.
ALGOS = {"dqn": DQN, "ppo": PPO}


def train(
    algo: str = "dqn",
    preset: str = "baseline",
    timesteps: int = DEFAULT_TIMESTEPS,
    run_name: str | None = None,
    seed: int = TRAIN_SEED,
) -> Path:
    """Entraine un agent et sauvegarde le resultat.

    Produit dans models/<run_name>/ :
        - best_model.zip : meilleur modele vu pendant l'entrainement
        - final.zip      : modele a la fin de l'entrainement
        - metadata.json  : algo, hyperparametres, duree (tracabilite)

    Retourne le repertoire du run.
    """
    if algo not in ALGOS:
        raise ValueError(f"Algo inconnu : {algo!r}. Attendu : {list(ALGOS)}")
    if preset not in PRESETS:
        raise ValueError(f"Preset inconnu : {preset!r}. Attendu : {list(PRESETS)}")

    run_name = run_name or f"{algo}_{preset}"
    run_dir = MODELS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    hyperparams = PRESETS[preset]

    train_env = make_env(seed=seed)
    # Environnement d'evaluation separe, avec une autre seed : evaluer sur
    # les memes niveaux que l'entrainement surestimerait la performance.
    eval_env = make_env(seed=EVAL_SEED)

    model = ALGOS[algo](
        "MlpPolicy",          # observation = 8 flottants -> un MLP suffit,
                              # pas besoin de couches convolutives
        train_env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(RUNS_DIR),
        **hyperparams,
    )

    # Evalue periodiquement et conserve le meilleur modele. Sans ca, on
    # garderait le modele final, qui n'est pas forcement le meilleur : les
    # performances d'un DQN oscillent en cours d'entrainement.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(timesteps // 20, 1000),
        n_eval_episodes=10,     # rapide : c'est un suivi, pas la mesure finale
        deterministic=True,
        verbose=0,
    )

    started = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        tb_log_name=run_name,
        progress_bar=False,
    )
    duration = time.time() - started

    model.save(run_dir / "final")

    metadata = {
        "run_name": run_name,
        "algo": algo,
        "preset": preset,
        "hyperparams": hyperparams,
        "timesteps": timesteps,
        "train_seed": seed,
        "duration_seconds": round(duration, 1),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    train_env.close()
    eval_env.close()

    print(f"[train] {run_name}: {timesteps} steps en {duration:.0f}s -> {run_dir}")
    return run_dir


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Entraine un agent Eagle-1")
    parser.add_argument("--algo", default="dqn", choices=list(ALGOS))
    parser.add_argument("--preset", default="baseline", choices=list(PRESETS))
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args()
    train(
        algo=args.algo,
        preset=args.preset,
        timesteps=args.timesteps,
        run_name=args.run_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
