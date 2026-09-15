"""Interface graphique : visualiser une partie jouee par l'agent.

Livrable de l'etape 3 du sujet : "Creer un GUI avec Streamlit ou Gradio pour
visualiser une partie jouee par l'agent (ex. : animation de l'atterrissage)".

Ce fichier n'importe volontairement ni gymnasium, ni torch, ni
stable-baselines3 : il demande a l'API de jouer un episode et se contente
d'afficher ce qu'elle renvoie. Un test automatise verifie cette propriete.

Lancement :
    uvicorn api.main:app --port 8000     # dans un premier terminal
    streamlit run app/gui.py             # dans un second
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client as api

st.set_page_config(page_title="Eagle-1 : pilotage", page_icon="rocket", layout="wide")

# Une couleur stable par action, reutilisee sur tous les graphiques : l'oeil
# apprend le code couleur une fois et le retrouve partout.
COULEURS_ACTIONS = {
    "rien": "#9aa0a6",
    "propulseur gauche": "#4c8bf5",
    "propulseur principal": "#f5a623",
    "propulseur droit": "#00a97f",
}

AXES_CIRCONSTANCE = {
    "inclinaison": "Inclinaison",
    "altitude": "Altitude",
    "descente": "Vitesse de descente",
    "position": "Position laterale",
}


def afficher_entete() -> dict | None:
    """Caracteristiques du modele servi. Retourne None si l'API est absente."""
    try:
        infos = api.modele()
    except api.ApiIndisponible as erreur:
        st.error(
            f"L'API ne repond pas sur {api.URL_API}. "
            "Demarre-la dans un autre terminal avec "
            "`uvicorn api.main:app --port 8000`."
        )
        st.caption(f"Detail technique : {erreur}")
        return None

    evaluation = infos["evaluation_100_episodes"]
    colonnes = st.columns(4)
    # delta_color="off" partout : le second argument de st.metric sert ici de
    # sous-titre, pas de variation. Sans cela Streamlit affiche une fleche
    # verte montante devant "dqn_optimise_300k", ce qui n'a aucun sens.
    colonnes[0].metric("Agent", infos["algo"].upper(), infos["run"], delta_color="off")
    colonnes[1].metric(
        "Recompense moyenne",
        f"{evaluation['recompense_moyenne']:.1f}",
        f"ecart-type {evaluation['ecart_type']:.1f}",
        delta_color="off",
    )
    colonnes[2].metric(
        "Taux de reussite", f"{evaluation['taux_reussite']:.0%}", "sur 100 episodes",
        delta_color="off",
    )
    colonnes[3].metric(
        "Seuil du sujet",
        f"{infos['seuil_reussite']:.0f}",
        "atteint" if evaluation["objectif_atteint"] else "non atteint",
        delta_color="off",
    )
    return infos


def afficher_trajectoire(pas: list[dict]) -> None:
    """Descente, recompense cumulee et actions au fil du temps."""
    donnees = pd.DataFrame(
        {
            "pas": [p["t"] for p in pas],
            "altitude": [p["observation"][1] for p in pas],
            "vitesse verticale": [p["observation"][3] for p in pas],
            "angle": [p["observation"][4] for p in pas],
            "action": [p["action_nom"] for p in pas],
            "recompense cumulee": pd.Series([p["recompense"] for p in pas]).cumsum(),
        }
    )

    gauche, droite = st.columns(2)

    with gauche:
        st.markdown("**Descente**")
        figure = go.Figure()
        figure.add_scatter(x=donnees["pas"], y=donnees["altitude"], name="altitude")
        figure.add_scatter(
            x=donnees["pas"], y=donnees["vitesse verticale"], name="vitesse verticale"
        )
        figure.add_scatter(x=donnees["pas"], y=donnees["angle"], name="angle")
        # Le sol est a y = 0 : ce repere rend la courbe d'altitude lisible.
        figure.add_hline(y=0, line_dash="dot", line_color="gray")
        figure.update_layout(
            height=300, margin=dict(t=10, b=10), xaxis_title="pas de temps"
        )
        st.plotly_chart(figure, use_container_width=True)

    with droite:
        st.markdown("**Recompense cumulee**")
        figure = px.line(donnees, x="pas", y="recompense cumulee")
        figure.add_hline(
            y=200, line_dash="dash", line_color="green",
            annotation_text="seuil de reussite",
        )
        figure.update_layout(
            height=300, margin=dict(t=10, b=10), xaxis_title="pas de temps"
        )
        st.plotly_chart(figure, use_container_width=True)

    st.markdown("**Actions choisies au cours du temps**")
    figure = px.scatter(
        donnees,
        x="pas",
        y="action",
        color="action",
        color_discrete_map=COULEURS_ACTIONS,
        category_orders={"action": list(COULEURS_ACTIONS)},
    )
    figure.update_traces(marker=dict(size=5))
    figure.update_layout(
        height=220, margin=dict(t=10, b=10), showlegend=False,
        xaxis_title="pas de temps", yaxis_title="",
    )
    st.plotly_chart(figure, use_container_width=True)


def afficher_decisions(resume: dict) -> None:
    """Repartition des actions selon la circonstance, pour cet episode.

    Repond a l'exigence de l'enonce : montrer "les decisions prises par l'IA
    en fonction du type de circonstance".
    """
    st.markdown("**Que fait l'agent selon la situation ?**")
    onglets = st.tabs(list(AXES_CIRCONSTANCE.values()))

    for onglet, famille in zip(onglets, AXES_CIRCONSTANCE):
        with onglet:
            lignes = [
                {"situation": situation, "action": action, "pas": nombre}
                for situation, repartition in resume[famille].items()
                for action, nombre in repartition.items()
                if nombre
            ]
            if not lignes:
                st.info("Aucune donnee pour cet axe.")
                continue

            donnees = pd.DataFrame(lignes)
            # On affiche des parts et non des effectifs : sinon les situations
            # rares (penche a gauche) seraient ecrasees par les frequentes.
            donnees["part"] = donnees["pas"] / donnees.groupby("situation")["pas"].transform("sum")

            figure = px.bar(
                donnees,
                x="part",
                y="situation",
                color="action",
                orientation="h",
                color_discrete_map=COULEURS_ACTIONS,
                category_orders={"action": list(COULEURS_ACTIONS)},
                hover_data={"pas": True, "part": ":.0%"},
            )
            figure.update_layout(
                height=280, margin=dict(t=10, b=10),
                xaxis_tickformat=".0%", xaxis_title="part des decisions",
                yaxis_title="", legend_title="",
            )
            st.plotly_chart(figure, use_container_width=True)


st.title("Eagle-1 : visualiser un atterrissage")
st.caption(
    "L'agent joue une partie de LunarLander-v3. Toute la simulation tourne dans "
    "l'API ; cette page ne fait qu'afficher ce qu'elle renvoie."
)

infos_modele = afficher_entete()

if infos_modele:
    st.divider()

    controles = st.columns([1, 1, 2])
    with controles[0]:
        seed_aleatoire = st.checkbox(
            "Seed aleatoire",
            value=False,
            help="Decoche pour rejouer exactement la meme partie a chaque fois.",
        )
    with controles[1]:
        seed = st.number_input("Seed", value=7, step=1, disabled=seed_aleatoire)
    with controles[2]:
        st.write("")
        lancer = st.button(
            "Faire atterrir Eagle-1", type="primary", use_container_width=True
        )

    if lancer:
        with st.spinner("Simulation et rendu video en cours..."):
            try:
                st.session_state["episode"] = api.jouer(
                    seed=None if seed_aleatoire else int(seed), avec_video=True
                )
            except api.ApiIndisponible as erreur:
                st.error(f"Echec de l'appel a l'API : {erreur}")

    episode = st.session_state.get("episode")

    if episode is None:
        st.info("Clique sur **Faire atterrir Eagle-1** pour lancer une partie.")
    else:
        st.divider()
        resultat = st.columns(4)
        resultat[0].metric("Recompense", f"{episode['recompense_totale']:.1f}")
        resultat[1].metric("Duree", f"{episode['nb_pas']} pas")
        resultat[2].metric(
            "Atterrissage", "reussi" if episode["reussi"] else "manque"
        )
        resultat[3].metric("Pieds au contact", "oui" if episode["pose"] else "non")

        colonne_video, colonne_decisions = st.columns([1, 1])

        with colonne_video:
            st.markdown("**Animation de l'atterrissage**")
            try:
                st.video(api.video(episode["episode_id"]))
            except Exception as erreur:  # noqa: BLE001 - degrader plutot que planter
                st.warning(f"Video indisponible : {erreur}")

        with colonne_decisions:
            afficher_decisions(episode["decisions_par_circonstance"])

        st.divider()
        afficher_trajectoire(api.episode(episode["episode_id"])["pas"])
