"""Tableau de bord : suivi interactif des performances de l'agent.

Livrable de l'etape 3 du sujet : "Construire un tableau de bord affichant
les courbes de recompense, moyenne, ecart-type, les decisions prises par
l'IA en fonction du type de circonstance".

Difference avec le GUI : le GUI montre UNE partie, ce tableau de bord agrege
TOUTES les parties jouees, et compare les douze experiences d'entrainement.

Comme le GUI, il n'importe ni gymnasium, ni torch, ni stable-baselines3 :
toutes ses donnees viennent de l'API par HTTP.

Lancement :
    uvicorn api.main:app --port 8000        # dans un premier terminal
    streamlit run app/dashboard.py          # dans un second
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client as api

st.set_page_config(page_title="Eagle-1 : tableau de bord", page_icon="chart", layout="wide")

SEUIL = 200.0

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


# --------------------------------------------------------------------------
# Onglet 1 : performance sur les parties jouees
# --------------------------------------------------------------------------
def onglet_parties(metriques: dict, episodes: list[dict]) -> None:
    if not episodes:
        st.info(
            "Aucune partie jouee pour l'instant. Utilise le panneau de gauche "
            "pour en lancer une serie."
        )
        return

    colonnes = st.columns(5)
    colonnes[0].metric("Parties jouees", metriques["nb_episodes"])
    colonnes[1].metric("Recompense moyenne", f"{metriques['recompense_moyenne']:.1f}")
    colonnes[2].metric("Ecart-type", f"{metriques['ecart_type']:.1f}")
    colonnes[3].metric("Taux de reussite", f"{metriques['taux_reussite']:.0%}")
    colonnes[4].metric("Duree moyenne", f"{metriques['duree_moyenne']:.0f} pas")

    # L'API renvoie les episodes du plus recent au plus ancien : on remet dans
    # l'ordre chronologique pour que la courbe se lise de gauche a droite.
    donnees = pd.DataFrame(reversed(episodes))
    donnees["numero"] = range(1, len(donnees) + 1)
    # Moyenne glissante : une courbe brute d'episodes est trop bruitee pour
    # qu'on y voie une tendance (ecart-type de ~78 sur le modele final).
    fenetre = max(min(len(donnees) // 5, 20), 2)
    donnees["moyenne glissante"] = donnees["recompense_totale"].rolling(fenetre, min_periods=1).mean()

    gauche, droite = st.columns([3, 2])

    with gauche:
        st.markdown("**Recompense par partie**")
        figure = go.Figure()
        figure.add_scatter(
            x=donnees["numero"], y=donnees["recompense_totale"],
            mode="markers", name="partie", marker=dict(size=6, color="#4c8bf5"),
        )
        figure.add_scatter(
            x=donnees["numero"], y=donnees["moyenne glissante"],
            mode="lines", name=f"moyenne glissante ({fenetre})",
            line=dict(color="#f5a623", width=3),
        )
        figure.add_hline(
            y=SEUIL, line_dash="dash", line_color="green",
            annotation_text="seuil de reussite (200)",
        )
        figure.add_hline(
            y=metriques["recompense_moyenne"], line_dash="dot", line_color="gray",
            annotation_text=f"moyenne {metriques['recompense_moyenne']:.0f}",
        )
        figure.update_layout(
            height=360, margin=dict(t=10, b=10),
            xaxis_title="partie", yaxis_title="recompense",
        )
        st.plotly_chart(figure, use_container_width=True)

    with droite:
        st.markdown("**Distribution des recompenses**")
        figure = px.histogram(donnees, x="recompense_totale", nbins=20)
        figure.add_vline(x=SEUIL, line_dash="dash", line_color="green")
        figure.update_layout(
            height=360, margin=dict(t=10, b=10),
            xaxis_title="recompense", yaxis_title="nombre de parties", bargap=0.05,
        )
        st.plotly_chart(figure, use_container_width=True)

    st.markdown("**Detail des parties**")
    tableau = donnees[
        ["numero", "cree_le", "seed", "recompense_totale", "nb_pas", "reussi", "pose"]
    ].copy()
    # La base stocke des valeurs techniques : seed nulle quand elle est tiree
    # au hasard, booleens en 0/1. On les traduit pour l'affichage.
    tableau["seed"] = tableau["seed"].map(lambda s: "aleatoire" if pd.isna(s) else int(s))
    for colonne in ("reussi", "pose"):
        tableau[colonne] = tableau[colonne].map({1: "oui", 0: "non"})
    tableau["cree_le"] = pd.to_datetime(tableau["cree_le"]).dt.strftime("%d/%m %H:%M:%S")
    tableau = tableau.rename(
        columns={
            "numero": "n", "cree_le": "date", "recompense_totale": "recompense",
            "nb_pas": "duree (pas)",
        }
    )
    st.dataframe(
        tableau.iloc[::-1].style.format({"recompense": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------------------------------
# Onglet 2 : decisions de l'agent par circonstance
# --------------------------------------------------------------------------
def onglet_decisions(metriques: dict) -> None:
    if not metriques["nb_episodes"]:
        st.info("Aucune donnee : joue d'abord quelques parties.")
        return

    st.markdown("**Repartition globale des actions**")
    repartition = pd.DataFrame(
        [{"action": a, "pas": n} for a, n in metriques["repartition_actions"].items()]
    )
    repartition["part"] = repartition["pas"] / repartition["pas"].sum()
    figure = px.bar(
        repartition, x="part", y="action", orientation="h", color="action",
        color_discrete_map=COULEURS_ACTIONS, text=repartition["part"].map("{:.0%}".format),
    )
    figure.update_layout(
        height=220, margin=dict(t=10, b=10), showlegend=False,
        xaxis_tickformat=".0%", xaxis_title="part des decisions", yaxis_title="",
    )
    st.plotly_chart(figure, use_container_width=True)

    st.divider()
    st.markdown("**Que fait l'agent selon la circonstance ?**")
    st.caption(
        "Chaque pas de temps est classe selon quatre axes. On lit ici la part "
        "de chaque action dans chaque situation, tous episodes confondus."
    )

    decisions = metriques["decisions_par_circonstance"]
    colonnes = st.columns(2)

    for index, (famille, titre) in enumerate(AXES_CIRCONSTANCE.items()):
        lignes = [
            {"situation": situation, "action": action, "pas": nombre}
            for situation, repartition in decisions[famille].items()
            for action, nombre in repartition.items()
            if nombre
        ]
        if not lignes:
            continue

        donnees = pd.DataFrame(lignes)
        # Parts et non effectifs : sinon les situations rares (penche a
        # gauche) seraient invisibles a cote des frequentes.
        donnees["part"] = donnees["pas"] / donnees.groupby("situation")["pas"].transform("sum")

        with colonnes[index % 2]:
            st.markdown(f"*{titre}*")
            figure = px.bar(
                donnees, x="part", y="situation", color="action", orientation="h",
                color_discrete_map=COULEURS_ACTIONS,
                category_orders={"action": list(COULEURS_ACTIONS)},
                hover_data={"pas": True, "part": ":.0%"},
            )
            figure.update_layout(
                height=250, margin=dict(t=10, b=30),
                xaxis_tickformat=".0%", xaxis_title="part des decisions",
                yaxis_title="", legend_title="",
            )
            st.plotly_chart(figure, use_container_width=True)


# --------------------------------------------------------------------------
# Onglet 3 : comparatif des experiences d'entrainement
# --------------------------------------------------------------------------
def onglet_experiences() -> None:
    donnees_api = api.experiences()
    experiences = donnees_api["experiences"]
    servi = donnees_api["modele_servi"]

    st.caption(
        f"Douze entrainements, chacun evalue sur 100 episodes. Le modele servi "
        f"par l'API est **{servi}**."
    )

    donnees = pd.DataFrame(experiences)
    donnees["sert"] = donnees["run"] == servi

    st.markdown("**Recompense moyenne par experience**")
    st.caption(
        "Les barres d'erreur montrent l'erreur-type : deux experiences dont les "
        "barres se chevauchent ne sont pas distinguables statistiquement."
    )
    figure = go.Figure()
    figure.add_bar(
        x=donnees["mean"], y=donnees["run"], orientation="h",
        error_x=dict(type="data", array=donnees["sem"], visible=True),
        marker_color=["#00a97f" if s else "#4c8bf5" for s in donnees["sert"]],
        hovertemplate="%{y}<br>moyenne %{x:.1f}<extra></extra>",
    )
    figure.add_vline(
        x=SEUIL, line_dash="dash", line_color="green",
        # En haut par defaut, l'annotation chevauchait la barre du meilleur
        # modele, qui est justement celle qui franchit le seuil.
        annotation_text="seuil 200", annotation_position="bottom right",
    )
    figure.update_layout(
        height=440, margin=dict(t=10, b=10),
        xaxis_title="recompense moyenne sur 100 episodes", yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(figure, use_container_width=True)

    gauche, droite = st.columns(2)

    with gauche:
        st.markdown("**Regularite : moyenne contre ecart-type**")
        st.caption(
            "En haut a gauche = bon et regulier. cible_rapide a le meilleur "
            "ecart-type de la campagne tout en etant le pire agent : il a appris "
            "le vol stationnaire sans jamais se poser."
        )
        # Etiqueter les douze points les ferait se chevaucher dans l'amas du
        # bas. On ne nomme que ceux qui portent une lecon : le modele servi,
        # le point de depart, et le contre-exemple du vol stationnaire. Les
        # autres restent lisibles au survol.
        remarquables = {"dqn_optimise_300k", "dqn_baseline", "dqn_cible_rapide"}
        donnees["etiquette"] = donnees["run"].where(donnees["run"].isin(remarquables), "")

        figure = px.scatter(
            donnees, x="std", y="mean", text="etiquette",
            size="success_rate", size_max=28, hover_name="run",
            hover_data={"mean": ":.1f", "std": ":.1f", "success_rate": ":.0%",
                        "etiquette": False},
            color=donnees["sert"].map({True: "modele servi", False: "autre"}),
            color_discrete_map={"modele servi": "#00a97f", "autre": "#4c8bf5"},
        )
        figure.update_traces(textposition="top center", textfont_size=10)
        figure.add_hline(y=SEUIL, line_dash="dash", line_color="green")
        figure.update_layout(
            height=420, margin=dict(t=10, b=10), legend_title="",
            xaxis_title="ecart-type (plus bas = plus regulier)",
            yaxis_title="recompense moyenne",
        )
        st.plotly_chart(figure, use_container_width=True)

    with droite:
        st.markdown("**Distribution des recompenses par experience**")
        st.caption(
            "Une boite etalee signale un agent imprevisible, meme si sa moyenne "
            "est correcte."
        )
        longues = pd.DataFrame(
            [
                {"run": ligne["run"], "recompense": valeur}
                for ligne in experiences
                for valeur in ligne["recompenses"]
            ]
        )
        ordre = donnees.sort_values("mean", ascending=False)["run"].tolist()
        figure = px.box(longues, x="recompense", y="run", category_orders={"run": ordre})
        figure.add_vline(x=SEUIL, line_dash="dash", line_color="green")
        figure.update_layout(
            height=420, margin=dict(t=10, b=10), xaxis_title="recompense", yaxis_title="",
        )
        st.plotly_chart(figure, use_container_width=True)

    st.markdown("**Detail des experiences**")
    tableau = donnees[
        ["run", "algo", "timesteps", "mean", "std", "sem", "success_rate",
         "crash_rate", "median", "objectif_atteint", "hyperparams"]
    ].rename(
        columns={
            "mean": "moyenne", "std": "ecart-type", "sem": "erreur-type",
            "success_rate": "reussite", "crash_rate": "crashes",
            "median": "mediane", "objectif_atteint": ">= 200",
            "hyperparams": "parametre modifie",
        }
    )
    tableau["parametre modifie"] = tableau["parametre modifie"].map(
        lambda h: ", ".join(f"{k}={v}" for k, v in h.items()) if h else "defauts SB3"
    )
    st.dataframe(
        tableau.style.format(
            {
                "moyenne": "{:.1f}", "ecart-type": "{:.1f}", "erreur-type": "{:.1f}",
                "mediane": "{:.1f}", "reussite": "{:.0%}", "crashes": "{:.0%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("Eagle-1 : tableau de bord")

try:
    infos_modele = api.modele()
except api.ApiIndisponible as erreur:
    st.error(
        f"L'API ne repond pas sur {api.URL_API}. Demarre-la dans un autre "
        "terminal avec `uvicorn api.main:app --port 8000`."
    )
    st.caption(f"Detail technique : {erreur}")
    st.stop()

evaluation = infos_modele["evaluation_100_episodes"]
st.caption(
    f"Agent servi : **{infos_modele['run']}** ({infos_modele['algo'].upper()}) — "
    f"{evaluation['recompense_moyenne']:.1f} ± {evaluation['ecart_type']:.1f} "
    f"sur 100 episodes d'evaluation, {evaluation['taux_reussite']:.0%} de reussite."
)

with st.sidebar:
    st.header("Alimenter le tableau")
    st.caption(
        "Chaque partie est jouee par l'API et enregistree en base. Les seeds "
        "sont aleatoires : c'est ce qui rend les statistiques representatives."
    )
    nombre = st.slider("Nombre de parties", 1, 30, 10)
    if st.button("Jouer", type="primary", use_container_width=True):
        barre = st.progress(0.0, "Simulation...")
        for index in range(nombre):
            try:
                api.jouer(seed=None, avec_video=False)
            except api.ApiIndisponible as erreur:
                st.error(f"Echec a la partie {index + 1} : {erreur}")
                break
            barre.progress((index + 1) / nombre, f"{index + 1}/{nombre} parties")
        barre.empty()
        st.rerun()

    st.divider()
    if st.button("Rafraichir", use_container_width=True):
        st.rerun()

metriques = api.metriques()
episodes = api.episodes(limite=500)["episodes"]

onglet1, onglet2, onglet3 = st.tabs(
    ["Parties jouees", "Decisions de l'agent", "Experiences d'entrainement"]
)

with onglet1:
    onglet_parties(metriques, episodes)
with onglet2:
    onglet_decisions(metriques)
with onglet3:
    onglet_experiences()
