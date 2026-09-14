"""Campagne d'experiences d'hyperparametres.

Enchaine entrainement + evaluation sur 100 episodes pour une liste de
presets, puis produit un tableau comparatif trie.

L'enonce impose de ne modifier qu'un seul hyperparametre a la fois : c'est
la definition des presets dans config.py qui garantit cette propriete, ce
module ne fait que les executer de facon identique et comparable.

Usage :
    python -m eagle1.experiments --presets net256 lr_haut cible_rapide
    python -m eagle1.experiments --tableau        # relit les resultats existants
"""

from __future__ import annotations

import argparse
import json

from .analysis import describe_distribution
from .config import EVAL_EPISODES_FINAL, MODELS_DIR, PRESETS, SUCCESS_THRESHOLD
from .evaluate import evaluate_run
from .train import train


def lancer(presets: list[str], algo: str = "dqn", timesteps: int = 100_000) -> None:
    """Entraine puis evalue chaque preset, dans les memes conditions."""
    for index, preset in enumerate(presets, start=1):
        run = f"{algo}_{preset}"
        print(f"\n[{index}/{len(presets)}] {run}", flush=True)
        train(algo=algo, preset=preset, timesteps=timesteps, run_name=run)
        stats = evaluate_run(run, n_episodes=EVAL_EPISODES_FINAL)
        print(
            f"    -> {stats['mean_reward']:7.1f} +/- {stats['std_reward']:5.1f}"
            f"   reussite {stats['success_rate']:.0%}",
            flush=True,
        )


def tableau(episodes: int = EVAL_EPISODES_FINAL) -> str:
    """Tableau comparatif de toutes les evaluations disponibles."""
    lignes = []
    for dossier in sorted(MODELS_DIR.iterdir()):
        fichier = dossier / f"eval_best_model_{episodes}ep.json"
        if not dossier.is_dir() or not fichier.exists():
            continue
        brut = json.loads(fichier.read_text(encoding="utf-8"))
        s = describe_distribution(brut["episode_rewards"])
        lignes.append((s["mean"], dossier.name, s))

    lignes.sort(reverse=True)
    entete = (
        f"{'run':<26} {'moyenne':>9} {'ecart-type':>11} {'erreur-type':>12} "
        f"{'>=200':>7} {'mediane':>9}"
    )
    corps = [
        f"{nom:<26} {s['mean']:9.1f} {s['std']:11.1f} {s['sem']:12.1f} "
        f"{s['success_rate']:6.0%} {s['median']:9.1f}"
        + ("  <= objectif atteint" if s["mean"] >= SUCCESS_THRESHOLD else "")
        for _, nom, s in lignes
    ]
    return "\n".join([entete, "-" * len(entete), *corps])


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Campagne d'hyperparametres")
    parser.add_argument("--presets", nargs="*", default=[p for p in PRESETS if p != "ppo_defaut"])
    parser.add_argument("--algo", default="dqn")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--tableau", action="store_true", help="affiche le comparatif sans rien entrainer")
    args = parser.parse_args()

    if not args.tableau:
        lancer(args.presets, algo=args.algo, timesteps=args.timesteps)
    print("\n" + tableau())


if __name__ == "__main__":
    _cli()
