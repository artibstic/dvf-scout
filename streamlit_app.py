from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from core import (
    build_argumentaire,
    enrich_location_flags,
    estimate_market,
    format_eur,
    geocode_address,
    load_city_dvf,
    prepare_mutations,
    select_comparables,
    street_summary,
)

st.set_page_config(page_title="DVF Scout", page_icon="🏠", layout="wide")

st.title("DVF Scout · V1")
st.caption("Analyse factuelle d'un marché immobilier à partir des données publiques DVF géolocalisées.")

with st.sidebar:
    st.header("Bien étudié")
    address = st.text_input("Adresse", value="", placeholder="Ex. 10 rue de la Paix, 75002 Paris")
    target_surface = st.number_input("Surface (m²)", min_value=10.0, max_value=1000.0, value=50.0, step=0.5)
    target_rooms = st.number_input("Pièces principales", min_value=0, max_value=20, value=2, step=1, help="0 = ne pas privilégier le nombre de pièces")
    target_type = st.selectbox("Type de bien", ["Appartement", "Maison"], index=0)

    st.header("Comparables")
    radius_m = st.slider("Rayon maximum", 100, 2000, 500, 50, format="%d m")
    surface_tol_pct = st.slider("Tolérance de surface", 10, 60, 25, 5, format="±%d %%")
    history_years = st.slider("Historique DVF", 2, 5, 5, 1, help="DVF diffuse les cinq dernières années disponibles.")

    st.header("Contexte qualitatif")
    qualitative = st.text_area(
        "Éléments non présents dans DVF",
        value="",
        height=110,
        help="Ces éléments sont repris dans l'argumentaire mais ne créent pas artificiellement une prime de prix.",
    )

    run = st.button("Analyser", type="primary", use_container_width=True)

st.info(
    "V1 : la valorisation est fondée uniquement sur des transactions publiques observées. "
    "Les critères qualitatifs servent à l'argumentaire, pas à inventer une majoration automatique."
)

if run and not address.strip():
    st.warning("Saisissez une adresse avant de lancer l’analyse.")
    st.stop()

if not run:
    st.subheader("Ce que fait cette version")
    st.markdown(
        """
        - géocode l'adresse via le service IGN/Géoplateforme ;
        - télécharge les fichiers DVF géolocalisés de la commune ;
        - reconstruit une ligne par mutation et écarte les ventes multibiens ambiguës ;
        - repère les ventes au même numéro, dans la même rue et à proximité ;
        - classe les comparables selon surface, distance, récence et nombre de pièces ;
        - calcule une médiane pondérée et une fourchette robuste ;
        - produit un argumentaire sourcé et exportable.
        """
    )
    st.stop()

try:
    with st.status("Analyse en cours…", expanded=True) as status:
        st.write("1. Géocodage de l'adresse")
        geo = geocode_address(address)
        st.write(f"Adresse retenue : **{geo.label}** · code commune {geo.citycode}")

        current_year = datetime.now().year
        candidate_years = list(range(current_year, current_year - history_years - 1, -1))
        st.write("2. Récupération des transactions DVF")
        raw, loaded_years = load_city_dvf(geo.citycode, candidate_years, max_loaded=history_years)
        st.write("Millésimes disponibles : " + ", ".join(map(str, loaded_years)))

        st.write("3. Reconstruction et nettoyage des mutations")
        mutations = prepare_mutations(raw, target_type=target_type)
        if mutations.empty:
            raise RuntimeError("Aucune vente exploitable pour ce type de bien.")
        mutations = enrich_location_flags(mutations, geo)

        st.write("4. Sélection des comparables")
        comps = select_comparables(
            mutations,
            target_surface=target_surface,
            target_rooms=int(target_rooms) if target_rooms else None,
            radius_m=radius_m,
            surface_tolerance=surface_tol_pct / 100,
        )
        if comps.empty:
            raise RuntimeError(
                "Aucun comparable n'a passé les filtres. Élargissez le rayon ou la tolérance de surface."
            )
        estimate = estimate_market(comps, target_surface)
        summary = street_summary(mutations)
        status.update(label="Analyse terminée", state="complete", expanded=False)

except Exception as exc:
    st.error(str(exc))
    st.stop()

st.success(f"{len(comps)} comparables fiables retenus pour {geo.label}")

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Niveau central", format_eur(estimate["prix_m2_central"], True))
c2.metric("Valeur indicative", format_eur(estimate["valeur_centrale"]))
c3.metric("Fourchette €/m²", f"{format_eur(estimate['prix_m2_bas'], True)} → {format_eur(estimate['prix_m2_haut'], True)}")
c4.metric("Score médian", f"{estimate['score_median']:.0f}/100")

if len(comps) < 5:
    st.warning("Échantillon réduit : la fourchette doit être interprétée avec prudence.")

# Same building / street
st.subheader("Même immeuble et même rue")
b1, b2 = st.columns(2)
bs = summary["building"]
ss = summary["street"]
with b1:
    st.markdown("**Même numéro**")
    if bs["n"]:
        st.write(f"{bs['n']} vente(s) simple(s) · médiane {format_eur(bs['median'], True)}")
    else:
        st.write("Aucune vente simple exploitable retrouvée au même numéro.")
with b2:
    st.markdown("**Même rue**")
    if ss["n"]:
        st.write(f"{ss['n']} vente(s) simple(s) · médiane {format_eur(ss['median'], True)}")
    else:
        st.write("Aucune vente simple exploitable retrouvée dans la même rue.")

# Charts
st.subheader("Lecture du marché")
chart_col, evol_col = st.columns(2)
with chart_col:
    fig = px.scatter(
        comps,
        x="surface_reelle_bati",
        y="prix_m2",
        size="score",
        hover_data=["adresse_complete", "date_mutation", "valeur_fonciere", "distance_m", "score"],
        labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix au m² (€)", "score": "Score"},
        title="Comparables retenus",
    )
    fig.add_vline(x=target_surface, line_dash="dash", annotation_text="Bien étudié")
    st.plotly_chart(fig, use_container_width=True)

with evol_col:
    yearly = comps.assign(annee=comps["date_mutation"].dt.year).groupby("annee", as_index=False)["prix_m2"].median()
    fig2 = px.line(yearly, x="annee", y="prix_m2", markers=True, labels={"annee": "Année", "prix_m2": "Médiane €/m²"}, title="Médiane des comparables par année")
    st.plotly_chart(fig2, use_container_width=True)

# Map
st.subheader("Carte des références")
map_df = comps[["latitude", "longitude"]].dropna().copy()
if not map_df.empty:
    st.map(map_df, latitude="latitude", longitude="longitude", zoom=14)

# Comparable table
st.subheader("Comparables classés")
display = comps[[
    "date_mutation", "adresse_complete", "surface_reelle_bati", "nombre_pieces_principales",
    "valeur_fonciere", "prix_m2", "distance_m", "score", "meme_numero", "meme_rue",
]].copy()
display.columns = ["Date", "Adresse", "Surface m²", "Pièces", "Prix", "€/m²", "Distance m", "Score", "Même numéro", "Même rue"]
display["Date"] = display["Date"].dt.date
display["Surface m²"] = display["Surface m²"].round(1)
display["Prix"] = display["Prix"].round(0).astype("Int64")
display["€/m²"] = display["€/m²"].round(0).astype("Int64")
display["Distance m"] = display["Distance m"].round(0).astype("Int64")
st.dataframe(display, use_container_width=True, hide_index=True)

# Argumentaire
st.subheader("Argumentaire factuel")
argument = build_argumentaire(geo.label, estimate, summary, comps, qualitative)
st.text_area("Texte prêt à copier", value=argument, height=360)

csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
report_bytes = argument.encode("utf-8")
d1, d2 = st.columns(2)
with d1:
    st.download_button("Télécharger les comparables (.csv)", data=csv_bytes, file_name="comparables_dvf.csv", mime="text/csv", use_container_width=True)
with d2:
    st.download_button("Télécharger l'argumentaire (.txt)", data=report_bytes, file_name="argumentaire_dvf.txt", mime="text/plain", use_container_width=True)

with st.expander("Méthode et limites"):
    st.markdown(
        """
        **Nettoyage.** Une mutation n'est retenue comme comparable principal que si elle contient un seul local principal identifiable du type étudié. Les mutations multibiens ambiguës sont écartées du calcul.

        **Score /100.** Surface (30), distance (25), récence (20), pièces (10), même rue (5), même numéro (10).

        **Estimation.** Le niveau central est une médiane pondérée par le score. La fourchette affichée correspond aux 25e et 75e percentiles de l'échantillon retenu.

        **Limites.** DVF ne décrit pas précisément l'étage, l'état intérieur, la vue, la qualité des travaux, l'exposition ou la situation locative. Cette V1 n'est donc ni une expertise immobilière réglementée ni une estimation notariale.
        """
    )

st.caption(
    "Sources : DGFiP / data.gouv.fr – Demandes de valeurs foncières géolocalisées ; IGN/Géoplateforme – géocodage."
)
