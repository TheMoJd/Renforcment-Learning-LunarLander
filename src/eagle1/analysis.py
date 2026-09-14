"""Analyse de la distribution des recompenses.

La moyenne seule ment. Le sujet exige un score >= 200 "stable, avec un
ecart-type faible" : ce module fournit les statistiques qui permettent de
le demontrer, et sert de source de donnees au tableau de bord.

Usage :
    python -m eagle1.analysis --run dqn_baseline --episodes 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import MODELS_DIR, SUCCESS_THRESHOLD


def describe_distribution(rewards: list[float] | np.ndarray) -> dict:
    """Statistiques de dispersion d'une serie de recompenses d'episodes."""
    values = np.asarray(rewards, dtype=float)
    n = len(values)

    reussis = values[values >= SUCCESS_THRESHOLD]
    rates = values[values < SUCCESS_THRESHOLD]

    return {
        "n_episodes": n,
        "mean": float(values.mean()),
        # Ecart-type : dispersion des EPISODES autour de la moyenne.
        # Repond a "mon agent est-il regulier ?"
        "std": float(values.std(ddof=1)),
        # Erreur-type : incertitude sur la MOYENNE elle-meme (std / racine(n)).
        # Repond a "puis-je faire confiance a ce chiffre de moyenne ?"
        # C'est elle qui dit si un ecart entre deux runs est significatif.
        "sem": float(values.std(ddof=1) / np.sqrt(n)),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
        "success_rate": float((values >= SUCCESS_THRESHOLD).mean()),
        # Un episode tres negatif est un crash : c'est ce qui detruit la moyenne.
        "crash_rate": float((values < -100).mean()),
        "mean_if_success": float(reussis.mean()) if len(reussis) else None,
        "mean_if_failure": float(rates.mean()) if len(rates) else None,
        # Intervalle de confiance a 95 % sur la moyenne.
        "ci95_low": float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(n)),
        "ci95_high": float(values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(n)),
    }


def taux_de_reussite_requis(stats: dict, cible: float = SUCCESS_THRESHOLD) -> float | None:
    """Part d'episodes reussis necessaire pour atteindre la moyenne cible.

    On resout  p * (moyenne des reussites) + (1-p) * (moyenne des echecs) = cible
    en gardant les niveaux de performance actuels. Cela transforme un objectif
    abstrait ("+110 points") en objectif actionnable ("moins crasher").
    """
    s, e = stats["mean_if_success"], stats["mean_if_failure"]
    if s is None or e is None or s == e:
        return None
    return (cible - e) / (s - e)


def histogramme_texte(rewards, n_bins: int = 12, largeur: int = 46) -> str:
    """Histogramme ASCII : rend la bimodalite visible d'un coup d'oeil."""
    values = np.asarray(rewards, dtype=float)
    effectifs, bornes = np.histogram(values, bins=n_bins)
    echelle = largeur / max(effectifs.max(), 1)

    lignes = []
    for i, effectif in enumerate(effectifs):
        bas, haut = bornes[i], bornes[i + 1]
        marqueur = " <- seuil 200" if bas <= SUCCESS_THRESHOLD < haut else ""
        lignes.append(
            f"  [{bas:7.0f} , {haut:7.0f}]  {'#' * int(effectif * echelle):<{largeur}} "
            f"{effectif:3d}{marqueur}"
        )
    return "\n".join(lignes)


def rapport(run: str, episodes: int, checkpoint: str = "best_model") -> str:
    """Rapport complet lisible, pour le notebook et la CLI."""
    chemin = MODELS_DIR / run / f"eval_{checkpoint}_{episodes}ep.json"
    if not chemin.exists():
        raise FileNotFoundError(
            f"{chemin} absent. Lance d'abord :\n"
            f"  python -m eagle1.evaluate --run {run} --episodes {episodes}"
        )

    rewards = json.loads(chemin.read_text(encoding="utf-8"))["episode_rewards"]
    s = describe_distribution(rewards)
    requis = taux_de_reussite_requis(s)

    return "\n".join(
        [
            f"Distribution des recompenses -- {run}/{checkpoint}, {s['n_episodes']} episodes",
            "",
            histogramme_texte(rewards),
            "",
            f"  Moyenne            {s['mean']:8.1f}",
            f"  Mediane            {s['median']:8.1f}   (si tres differente de la moyenne,",
            f"                                la distribution est asymetrique)",
            f"  Ecart-type         {s['std']:8.1f}   dispersion des episodes",
            f"  Erreur-type        {s['sem']:8.1f}   incertitude sur la moyenne",
            f"  IC 95 % moyenne    [{s['ci95_low']:.1f} ; {s['ci95_high']:.1f}]",
            "",
            f"  Quartiles Q1 / Q3  {s['q1']:8.1f} / {s['q3']:.1f}",
            f"  Min / Max          {s['min']:8.1f} / {s['max']:.1f}",
            "",
            f"  Episodes >= 200    {s['success_rate']:7.0%}   moyenne quand ca marche : {s['mean_if_success']:.1f}",
            f"  Crashes (< -100)   {s['crash_rate']:7.0%}   moyenne quand ca rate   : {s['mean_if_failure']:.1f}",
            "",
            (
                f"  Pour atteindre 200 de moyenne sans rien changer d'autre, il faudrait"
                f"\n  faire passer le taux de reussite de {s['success_rate']:.0%} a {requis:.0%}."
                if requis is not None
                else ""
            ),
        ]
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Analyse la dispersion des resultats")
    parser.add_argument("--run", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", default="best_model")
    args = parser.parse_args()
    print(rapport(args.run, args.episodes, args.checkpoint))


if __name__ == "__main__":
    _cli()
