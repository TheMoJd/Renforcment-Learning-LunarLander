"""Client HTTP de l'API Eagle-1, partage par le GUI et le tableau de bord.

Ce module est la SEULE porte d'entree du frontend vers le backend. Il ne
contient aucune logique d'apprentissage par renforcement : il ne sait ni ce
qu'est un environnement Gymnasium, ni ce qu'est un reseau de neurones. Il
envoie des requetes HTTP et lit du JSON.

C'est ce qui garantit la contrainte du sujet : "toute la logique RL doit
etre cote API (backend) et non cote GUI (frontend)".
"""

from __future__ import annotations

import os

import requests

# Surchargeable par variable d'environnement, pour pointer vers une API
# lancee sur un autre port sans toucher au code.
URL_API = os.environ.get("EAGLE1_API", "http://localhost:8000")

# Jouer un episode avec rendu video prend plusieurs dizaines de secondes :
# le delai par defaut de requests (aucun) serait ici un piege silencieux.
DELAI = 180


class ApiIndisponible(RuntimeError):
    """L'API ne repond pas.

    Exception dediee pour que l'interface affiche un message utile au lieu
    de planter sur une trace technique.
    """


def _get(chemin: str, **params) -> requests.Response:
    try:
        reponse = requests.get(f"{URL_API}{chemin}", params=params, timeout=DELAI)
    except requests.RequestException as erreur:
        raise ApiIndisponible(str(erreur)) from erreur
    reponse.raise_for_status()
    return reponse


def sante() -> dict:
    return _get("/health").json()


def modele() -> dict:
    return _get("/model").json()


def metriques() -> dict:
    return _get("/metrics").json()


def experiences() -> dict:
    return _get("/experiments").json()


def episodes(limite: int = 200) -> dict:
    return _get("/episodes", limite=limite).json()


def episode(episode_id: str) -> dict:
    return _get(f"/episodes/{episode_id}").json()


def video(episode_id: str) -> bytes:
    return _get(f"/episodes/{episode_id}/video").content


def jouer(seed: int | None = None, avec_video: bool = True) -> dict:
    """Demande a l'API de jouer un episode complet."""
    try:
        reponse = requests.post(
            f"{URL_API}/episodes",
            json={"seed": seed, "video": avec_video},
            timeout=DELAI,
        )
    except requests.RequestException as erreur:
        raise ApiIndisponible(str(erreur)) from erreur
    reponse.raise_for_status()
    return reponse.json()
