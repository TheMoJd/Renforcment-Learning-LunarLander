"""Tests des contraintes imposees par le sujet.

Ces tests ne verifient pas la qualite du code mais le respect du cahier des
charges. Ils doivent rester verts jusqu'a la soutenance.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eagle1.config import ENV_ID, EVAL_EPISODES_FINAL, SUCCESS_THRESHOLD
from eagle1.env import describe_env, make_env

ROOT = Path(__file__).resolve().parents[1]

# Contrainte n1 du sujet : "Toute la logique RL doit etre cote API (backend)
# et non cote GUI (frontend)."
BACKEND_ONLY_IMPORTS = ("gymnasium", "stable_baselines3", "torch")


@pytest.mark.parametrize("module", BACKEND_ONLY_IMPORTS)
def test_le_frontend_ne_contient_aucune_logique_rl(module: str) -> None:
    """app/ ne parle que HTTP a l'API : aucun import RL ne doit y figurer."""
    pattern = re.compile(rf"^\s*(import|from)\s+{module}\b", re.MULTILINE)
    fautifs = [
        fichier.relative_to(ROOT)
        for fichier in (ROOT / "app").rglob("*.py")
        if pattern.search(fichier.read_text(encoding="utf-8"))
    ]
    assert not fautifs, (
        f"{module} importe dans le frontend : {fautifs}. "
        "La logique RL doit vivre dans l'API."
    )


def test_environnement_conforme_au_sujet() -> None:
    """8 observations continues, 4 actions discretes, environnement v3."""
    description = describe_env()
    assert description["env_id"] == ENV_ID == "LunarLander-v3"
    assert description["observation_dim"] == 8
    assert description["n_actions"] == 4


def test_le_seuil_de_reussite_est_celui_du_sujet() -> None:
    assert SUCCESS_THRESHOLD == 200.0
    assert EVAL_EPISODES_FINAL == 100


def test_une_seed_donnee_produit_toujours_le_meme_episode() -> None:
    """Reproductibilite : sans elle, aucune comparaison d'experience n'a de sens."""

    def rejouer(seed: int) -> float:
        env = make_env(seed=seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        env.close()
        return total

    assert rejouer(7) == rejouer(7)
