# Eagle-1 — Pilote automatique d'alunissage

Agent d'apprentissage par renforcement qui pose le module **Eagle-1** sur une zone
cible lunaire, entraîné sur l'environnement Gymnasium `LunarLander-v3` et exposé via
une API locale, une interface graphique et un tableau de bord.

> Projet réalisé dans le cadre d'une mission OpenClassrooms (parcours ingénieur IA).

---

## Objectif

Obtenir une **récompense moyenne ≥ 200 sur 100 épisodes d'évaluation**, avec un
écart-type faible — seuil au-delà duquel un atterrissage est considéré comme réussi.

---

## Architecture

```
src/eagle1/     coeur RL, partagé par le notebook ET l'API (aucune duplication)
  config.py     constantes, chemins, presets d'hyperparamètres
  env.py        construction de l'environnement (point d'entrée unique)
  train.py      entraînement DQN / PPO + logs TensorBoard
  evaluate.py   évaluation via evaluate_policy (moyenne, écart-type, taux de succès)
api/            API FastAPI — joue les épisodes, expose les métriques
app/            Streamlit — GUI de visualisation + tableau de bord
notebooks/      démarche complète (exploration, choix, entraînement, tuning, éval)
tests/          tests des contraintes du cahier des charges
```

### Décision d'architecture : l'environnement tourne côté serveur

Le sujet impose que **toute la logique RL vive dans le backend**. Une lecture naïve
suggérerait un endpoint `/play` recevant un état et renvoyant une action — mais un tel
contrat obligerait le frontend à faire tourner l'environnement Gymnasium lui-même,
donc à héberger de la logique RL.

L'API joue donc les épisodes de bout en bout et le frontend n'est qu'un lecteur de
trajectoires. Cette contrainte est vérifiée automatiquement par un test : aucun fichier
de `app/` ne peut importer `gymnasium`, `stable_baselines3` ou `torch`.

### Décision : DQN comme algorithme principal

`LunarLander-v3` a un espace d'actions **discret** (4 propulseurs), ce qui désigne DQN.
PPO est entraîné en parallèle comme point de comparaison documenté.

*Note* : le brief affirme que PPO serait « optimisé pour les champs d'actions
continues ». C'est inexact ici — l'espace est discret, et PPO gère très bien le discret.
La comparaison est faite sur des mesures, pas sur cette affirmation.

---

## Installation

Prérequis : Python ≥ 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pip install -e .
```

`requirements.txt` liste les contraintes lisibles ; `requirements.lock.txt` contient
les versions exactes validées, pour reproduire l'environnement à l'identique.

---

## Utilisation

```bash
# Entraîner un agent
python -m eagle1.train --algo dqn --preset baseline --timesteps 100000

# Évaluer sur 100 épisodes
python -m eagle1.evaluate --run dqn_baseline --episodes 100

# Suivre les courbes d'entraînement
tensorboard --logdir runs/

# Lancer l'API
uvicorn api.main:app --port 8000

# Lancer le GUI et le tableau de bord
streamlit run app/gui.py
streamlit run app/dashboard.py

# Tests
pytest
```

---

## Reproductibilité

- Toutes les seeds sont fixées dans `src/eagle1/config.py`.
- La seed d'évaluation (`EVAL_SEED = 1000`) diffère volontairement de celle
  d'entraînement (`TRAIN_SEED = 42`) : on mesure la généralisation, pas la
  mémorisation des niveaux déjà vus.
- Chaque entraînement écrit un `metadata.json` (algorithme, hyperparamètres, durée).

---

## Résultats

<!-- RESULTATS -->
### Étape 1 — Référence de départ

Protocole identique pour les trois lignes : `evaluate_policy`, 50 épisodes,
seed d'évaluation 1000 (différente de l'entraînement), politique déterministe.

| Politique | Récompense moyenne | Écart-type | Épisodes ≥ 200 |
|---|---:|---:|---:|
| Aléatoire (plancher) | −210,6 | 104,3 | 0 % |
| DQN, hyperparamètres par défaut | **+90,0** | 168,5 | 38 % |
| *Objectif* | *≥ 200* | *faible* | — |

**Lecture.** L'agent a bel et bien appris : +300 points par rapport à une politique
aléatoire, et il réussit déjà 38 % de ses atterrissages. Mais il est **instable** —
un écart-type de 168,5 pour une moyenne de 90 signifie que les épisodes oscillent
entre le crash (−281) et l'atterrissage propre (+263). C'est précisément ce que
l'étape suivante doit corriger : le sujet exige une moyenne *stable*, pas une moyenne
obtenue en compensant des crashes par de bons épisodes.

Entraînement : 100 000 pas, 210 s sur CPU.

