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


def resultats(episodes: int = EVAL_EPISODES_FINAL) -> list[dict]:
    """Resultats structures de toutes les experiences evaluees.

    Sert a la fois au tableau texte ci-dessous et a la route /experiments de
    l'API, que le tableau de bord consomme. Le frontend n'a pas le droit de
    lire models/ directement : il ne parle que HTTP.
    """
    sorties = []
    for dossier in sorted(MODELS_DIR.iterdir()):
        if not dossier.is_dir():
            continue
        fichier = dossier / f"eval_best_model_{episodes}ep.json"
        metadonnees = dossier / "metadata.json"
        if not fichier.exists():
            continue

        brut = json.loads(fichier.read_text(encoding="utf-8"))
        stats = describe_distribution(brut["episode_rewards"])
        meta = json.loads(metadonnees.read_text(encoding="utf-8")) if metadonnees.exists() else {}

        sorties.append(
            {
                "run": dossier.name,
                "algo": meta.get("algo"),
                "preset": meta.get("preset"),
                # Preset vide = valeurs par defaut de SB3 : on le dit
                # explicitement plutot que d'afficher un tiret.
                "hyperparams": meta.get("hyperparams") or {},
                "timesteps": meta.get("timesteps"),
                "duree_entrainement_s": meta.get("duration_seconds"),
                "objectif_atteint": stats["mean"] >= SUCCESS_THRESHOLD,
                "recompenses": brut["episode_rewards"],
                **{c: stats[c] for c in (
                    "n_episodes", "mean", "std", "sem", "median",
                    "q1", "q3", "min", "max", "success_rate", "crash_rate",
                )},
            }
        )

    sorties.sort(key=lambda ligne: ligne["mean"], reverse=True)
    return sorties


def tableau(episodes: int = EVAL_EPISODES_FINAL) -> str:
    """Tableau comparatif de toutes les evaluations disponibles."""
    lignes = [(ligne["mean"], ligne["run"], ligne) for ligne in resultats(episodes)]
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
