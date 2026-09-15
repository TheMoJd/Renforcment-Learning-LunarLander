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

# --- Modele retenu ------------------------------------------------------
# Agent servi par l'API, le GUI et la video. Mesure sur 100 episodes et
# quatre seeds d'evaluation : 239,0 / 246,7 / 240,0 / 252,1.
BEST_RUN = "dqn_optimise_300k"

# --- Presets d'hyperparametres -----------------------------------------
# Chaque preset = une experience nommee, tracable dans TensorBoard et dans
# le notebook. "baseline" est volontairement vide : il utilise les valeurs
# par defaut de Stable-Baselines3, ce qui est le point de depart demande
# a l'etape 1.
PRESETS: dict[str, dict] = {
    # --- Point de depart ------------------------------------------------
    # Vide = valeurs par defaut de Stable-Baselines3 :
    #   learning_rate 1e-4, buffer_size 1e6, batch_size 32, net_arch [64,64],
    #   target_update_interval 10000, exploration_fraction 0.1
    # Mesure : 72,6 +/- 173,3 sur 100 episodes, 34 % de reussite.
    "baseline": {},

    # --- Experiences : UN SEUL parametre change a la fois ----------------
    # Le baseline utilise deux couches de 64 neurones. LunarLander demande
    # de coordonner 8 variables continues vers un geste precis : on teste si
    # la capacite du reseau est le facteur limitant.
    "net256": {"policy_kwargs": {"net_arch": [256, 256]}},

    # 1e-4 est prudent. Avec seulement 100 000 pas, un pas d'apprentissage
    # plus grand peut etre necessaire pour converger dans le budget.
    "lr_haut": {"learning_rate": 6.3e-4},

    # Le reseau cible n'est rafraichi que tous les 10 000 pas par defaut :
    # l'agent poursuit une cible tres perimee. Le rafraichir plus souvent
    # devrait accelerer et stabiliser l'apprentissage.
    "cible_rapide": {"target_update_interval": 250},

    # Un buffer de 1 million sur 100 000 pas ne se remplit jamais : l'agent
    # rejoue en permanence ses tres mauvais debuts. Un buffer plus petit
    # oublie les erreurs de jeunesse.
    "buffer_court": {"buffer_size": 50_000},

    # Des lots plus grands donnent des gradients moins bruites, donc des
    # mises a jour plus stables -- ce que vise l'etape 2.
    "lot_128": {"batch_size": 128},

    # Prolonge la phase d'exploration (10 % -> 12 % de l'entrainement).
    "exploration_longue": {"exploration_fraction": 0.12},

    # gamma pondere les recompenses futures : avec 0,99 une recompense
    # obtenue dans 100 pas vaut encore 37 % de sa valeur. L'enonce cite
    # nommement ce parametre parmi ceux a faire varier.
    # Plus myope : l'agent privilegie le gain immediat. Risque de se poser
    # trop vite sans soigner l'approche.
    "gamma_myope": {"gamma": 0.95},
    # Plus prevoyant : il valorise davantage l'atterrissage final, au risque
    # de tolerer de longues manoeuvres couteuses en carburant.
    "gamma_prevoyant": {"gamma": 0.999},

    # --- Combinaison des variantes gagnantes ----------------------------
    # A n'entrainer qu'apres analyse des experiences individuelles.
    "optimise": {
        "policy_kwargs": {"net_arch": [256, 256]},
        "learning_rate": 6.3e-4,
        "target_update_interval": 250,
        "buffer_size": 50_000,
        "batch_size": 128,
        "learning_starts": 0,
        "gradient_steps": -1,
        "exploration_fraction": 0.12,
        "exploration_final_eps": 0.1,
    },

    # --- Comparatif d'algorithme (a lancer avec --algo ppo) --------------
    # PPO gere aussi bien le discret que le continu. Sert a verifier
    # empiriquement le choix de DQN plutot qu'a le supposer.
    "ppo_defaut": {},
}
