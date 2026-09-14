"""Constantes, chemins et presets d'hyperparametres du projet Eagle-1.

Regle du projet : tout parametre susceptible de changer d'une experience a
l'autre vit ICI, jamais en dur dans un script. C'est ce qui rend les runs
reproductibles et comparables -- exigence explicite de l'enonce
("ne modifier qu'un seul hyperparametre a la fois").
"""

from pathlib import Path

# --- Environnement ------------------------------------------------------
# Le PDF du brief mentionne LunarLander-v2, retire depuis Gymnasium 1.0.
# v3 est le meme environnement (memes espaces, meme reward), renumerote.
ENV_ID = "LunarLander-v3"

# Seuil d'atterrissage reussi impose par le sujet.
SUCCESS_THRESHOLD = 200.0

# --- Reproductibilite ---------------------------------------------------
TRAIN_SEED = 42
# Seed d'evaluation VOLONTAIREMENT differente de celle d'entrainement :
# on veut mesurer la generalisation, pas la capacite a rejouer les niveaux
# deja vus. L'enonce demande explicitement de verifier le surapprentissage.
EVAL_SEED = 1000

# --- Protocole d'evaluation (impose par l'enonce) -----------------------
EVAL_EPISODES_BASELINE = 50    # etape 1 : "au moins 50 episodes"
EVAL_EPISODES_FINAL = 100      # etape 2 : "100 episodes d'evaluation"

# --- Budget d'entrainement ----------------------------------------------
# ~700 steps/s en CPU sur cette machine -> 100k steps ~= 2.5 min.
DEFAULT_TIMESTEPS = 100_000

# --- Chemins ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"          # logs TensorBoard
DATA_DIR = ROOT / "data"          # episodes.db (SQLite)
VIDEOS_DIR = ROOT / "videos"

for _directory in (MODELS_DIR, RUNS_DIR, DATA_DIR, VIDEOS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --- Presets d'hyperparametres -----------------------------------------
# Chaque preset = une experience nommee, tracable dans TensorBoard et dans
# le notebook. "baseline" est volontairement vide : il utilise les valeurs
# par defaut de Stable-Baselines3, ce qui est le point de depart demande
# a l'etape 1.
PRESETS: dict[str, dict] = {
    "baseline": {},
}
