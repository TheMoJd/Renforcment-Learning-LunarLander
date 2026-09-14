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

Protocole : `evaluate_policy`, politique déterministe, seed d'évaluation 1000
(différente de la seed d'entraînement 42, afin de mesurer la généralisation).

| Politique | Moyenne | Écart-type | Épisodes ≥ 200 |
|---|---:|---:|---:|
| Aléatoire (plancher, 50 ép.) | −210,6 | 104,3 | 0 % |
| **DQN, hyperparamètres par défaut** (100 ép.) | **+72,6** | **173,3** | 34 % |
| *Objectif* | *≥ 200* | *faible* | *≈ 89 %* |

Entraînement : 100 000 pas, 210 s sur CPU.

#### Distribution des récompenses

```
  [   -281 ,    -100]  ##############################   28 épisodes  (crashes)
  [   -100 ,      36]  ########                          8
  [     36 ,      82]                                    0   <- la moyenne tombe ici
  [     82 ,     172]  ################                 16
  [    172 ,     263]  ##################################################  48  (atterrissages)
```

| Statistique | Valeur | Lecture |
|---|---:|---|
| Moyenne | 72,6 | |
| Médiane | 165,2 | 93 points au-dessus de la moyenne : distribution asymétrique |
| Écart-type | 173,3 | dispersion des épisodes — l'agent est **irrégulier** |
| Erreur-type | 17,3 | incertitude sur la moyenne elle-même |
| IC 95 % | [38,7 ; 106,6] | |
| Q1 / Q3 | −113,1 / 213,1 | la moitié centrale va du crash à la réussite |

**Analyse.** L'agent a nettement appris : +283 points sur une politique aléatoire, et
34 % d'atterrissages réussis. Mais sa distribution est **bimodale** — il ne produit
presque jamais de résultat moyen, il réussit franchement ou il s'écrase. La tranche
qui contient la moyenne (36 à 82) est vide : aucun épisode ne ressemble à la moyenne.

**Conséquence pour l'étape suivante.** En conservant les niveaux de performance actuels
(226,2 en cas de réussite, −6,5 en cas d'échec), le taux de réussite nécessaire pour
atteindre une moyenne de 200 se calcule directement :

```
p × 226,2 + (1 − p) × (−6,5) = 200   →   p = 89 %
```

L'objectif de l'optimisation n'est donc pas « gagner 127 points de moyenne » mais
**faire passer le taux de réussite de 34 % à 89 %**, c'est-à-dire supprimer les crashes.

#### Note méthodologique : le bruit d'évaluation

Le même modèle, évalué sur 30 épisodes avec quatre seeds différentes, donne 117,7 /
51,0 / 61,3 / 32,5 — soit 85 points d'écart. Avec un écart-type de 173, l'incertitude
sur la moyenne vaut 173/√n : environ 32 points sur 30 épisodes, 17 sur 100.

C'est pourquoi l'évaluation finale se fait sur 100 épisodes, et pourquoi un écart de
moins de ~35 points entre deux configurations d'hyperparamètres ne peut pas être
considéré comme significatif.

Reproduire ces chiffres :

```bash
python -m eagle1.evaluate --run dqn_baseline --episodes 100
python -m eagle1.analysis --run dqn_baseline --episodes 100
```

### Étape 2 — Optimisation des hyperparamètres

Protocole imposé : une seule variable modifiée à la fois par rapport au baseline.
Chaque configuration est entraînée sur 100 000 pas puis évaluée sur 100 épisodes.

| Configuration | Moyenne | Écart-type | ≥ 200 | Crashes |
|---|---:|---:|---:|---:|
| **`optimise` — 300 000 pas** | **239,0** | **78,0** | **87 %** | 2 % |
| `optimise` — 100 000 pas | 166,8 | 63,7 | 40 % | 0 % |
| PPO par défaut — 300 000 pas | 86,5 | 145,5 | 38 % | 12 % |
| `baseline` (défauts SB3) | 72,6 | 173,3 | 34 % | 28 % |
| `exploration_longue` | 62,3 | 140,1 | 14 % | 15 % |
| `buffer_court` | 35,3 | 153,4 | 11 % | 20 % |
| `lr_haut` | −7,9 | 164,4 | 17 % | 48 % |
| `lot_128` | −75,7 | 160,7 | 0 % | 39 % |
| `net256` | −79,1 | 145,9 | 4 % | 54 % |
| `cible_rapide` | −89,5 | 25,9 | 0 % | 36 % |

#### Résultat principal : les hyperparamètres interagissent

**Les six modifications isolées dégradent toutes l'agent.** Pourtant leur combinaison
le fait passer de 72,6 à 166,8, puis à 239,0 avec un budget d'entraînement triplé.

Exemple le plus net : rafraîchir le réseau cible tous les 250 pas au lieu de 10 000
est la pire modification isolée (−89,5), parce que l'agent poursuit alors une cible
qui bouge sans cesse. Combinée à un `batch_size` de 128 — qui réduit le bruit des
gradients — elle devient bénéfique.

Cela met en évidence la limite du protocole « un paramètre à la fois » : il n'explore
que les axes autour du point de départ et ne peut pas atteindre un optimum situé en
diagonale. Le protocole reste utile pour *isoler* l'effet de chaque variable, mais il
n'est pas une méthode d'optimisation.

#### Un écart-type faible n'est pas un objectif en soi

`cible_rapide` affiche le plus faible écart-type de toute la campagne (25,9). C'est
pourtant le pire agent : sa durée moyenne d'épisode est de 1000 pas, soit exactement
la limite de troncature. **Il a appris à rester en vol stationnaire sans jamais se
poser** — parfaitement régulier à ne rien faire. La stabilité ne vaut que couplée à
une moyenne élevée.

#### DQN vs PPO

À budget égal (300 000 pas), DQN optimisé atteint 239,0 quand PPO par défaut plafonne
à 86,5. Le choix de DQN, motivé par l'espace d'actions discret, est donc confirmé
empiriquement — et non pas seulement supposé. PPO n'a pas bénéficié du même effort
d'optimisation, ce que cette comparaison ne préjuge pas.

#### Validation finale

Le seuil des 200 est franchi. Pour écarter l'hypothèse d'une seed favorable, le modèle
retenu a été réévalué sur 100 épisodes avec quatre seeds d'évaluation distinctes :

| Seed | Moyenne | Écart-type | IC 95 % | ≥ 200 |
|---:|---:|---:|---|---:|
| 1000 | 239,0 | 78,0 | [223,8 ; 254,3] | 87 % |
| 2000 | 246,7 | 60,7 | [234,8 ; 258,6] | 86 % |
| 3000 | 240,0 | 85,7 | [223,2 ; 256,8] | 89 % |
| 4000 | 252,1 | 42,4 | [243,7 ; 260,4] | 90 % |

**Moyenne des quatre campagnes : 244,5 ; plus faible campagne : 239,0.** Toutes les
bornes inférieures d'intervalle de confiance dépassent largement 200.

Progression d'ensemble : les crashes passent de 28 % à 2 %, l'écart-type de 173,3 à
78,0, et la durée moyenne d'un épisode de 625 à 328 pas — l'agent se pose désormais
deux fois plus vite, ce qui réduit d'autant la consommation de carburant.

```bash
python -m eagle1.train --algo dqn --preset optimise --timesteps 300000 --run-name dqn_optimise_300k
python -m eagle1.evaluate --run dqn_optimise_300k --episodes 100
python -m eagle1.experiments --tableau
```
