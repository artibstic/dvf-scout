from __future__ import annotations

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core import (
    analysis_confidence,
    build_argumentaire,
    build_takeaways,
    enrich_location_flags,
    estimate_market,
    fetch_dpe_candidates,
    format_eur,
    geocode_address,
    load_city_dvf,
    prepare_mutations,
    select_best_dpe,
    select_comparables,
    street_summary,
)

st.set_page_config(page_title="DVF Scout", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 1.8rem; padding-bottom: 4rem;}
    [data-testid="stHeader"] {background: rgba(255,255,255,0.88); backdrop-filter: blur(8px);}
    .hero {
        padding: 1.65rem 1.8rem;
        border: 1px solid rgba(49, 73, 102, 0.14);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(245,248,252,0.98), rgba(237,244,249,0.92));
        box-shadow: 0 10px 32px rgba(28, 44, 64, 0.06);
        margin-bottom: 1.2rem;
    }
    .hero-kicker {font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; opacity: .58;}
    .hero-title {font-size: 2.25rem; line-height: 1.05; font-weight: 760; margin: .25rem 0 .35rem 0;}
    .hero-sub {font-size: 1.02rem; opacity: .72; max-width: 820px;}
    .section-title {font-size: 1.45rem; font-weight: 720; margin-top: 1.8rem; margin-bottom: .25rem;}
    .section-sub {font-size: .92rem; opacity: .66; margin-bottom: .9rem;}
    .soft-card {
        border: 1px solid rgba(49, 73, 102, 0.12);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(250,252,254,0.92);
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 73, 102, 0.12);
        border-radius: 17px;
        padding: .95rem 1rem;
        background: rgba(250,252,254,0.95);
        min-height: 116px;
        box-shadow: 0 6px 18px rgba(28, 44, 64, 0.035);
    }
    div[data-testid="stMetricLabel"] {font-size: .83rem; opacity: .68;}
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    .dpe-wrap {display:flex; gap:18px; align-items:center; flex-wrap:wrap;}
    .dpe-badge {
        width: 66px; height: 66px; border-radius: 16px; display:flex; align-items:center; justify-content:center;
        color:#fff; font-size: 2rem; font-weight:800; box-shadow:0 8px 20px rgba(0,0,0,.10);
    }
    .dpe-A {background:#178b52;} .dpe-B {background:#55a944;} .dpe-C {background:#b2bc31;}
    .dpe-D {background:#e4b62d;} .dpe-E {background:#e58a2e;} .dpe-F {background:#d85a35;} .dpe-G {background:#a83c34;}
    .dpe-ND {background:#6b7280;}
    .takeaway {
        border-left: 4px solid #55708b; padding: .55rem .8rem; margin: .4rem 0; background: rgba(245,248,251,.82); border-radius: 0 10px 10px 0;
    }
    .tiny-note {font-size:.82rem; opacity:.68;}
    .muted-chip {display:inline-block; border:1px solid rgba(49,73,102,.15); border-radius:999px; padding:.24rem .6rem; margin:.1rem .2rem .1rem 0; font-size:.78rem; background:#f7f9fb;}
    .stButton > button[kind="primary"] {border-radius: 12px; min-height: 46px; font-weight: 700;}
    .stDownloadButton > button {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_load_city_dvf(citycode: str, years: tuple[int, ...], max_loaded: int):
    return load_city_dvf(citycode, years, max_loaded=max_loaded)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cached_dpe(geo_tuple: tuple, target_surface: float, target_type: str):
    # Rebuild a lightweight object accepted by core without caching a dataclass instance directly.
    from core import GeocodeResult

    geo = GeocodeResult(*geo_tuple)
    return fetch_dpe_candidates(geo, target_surface, target_type)


def geo_to_tuple(geo):
    return (
        geo.label, geo.longitude, geo.latitude, geo.citycode, geo.postcode,
        geo.housenumber, geo.street, geo.city, geo.score, geo.ban_id,
    )


def dpe_badge(letter: str) -> str:
    cls = str(letter or "").upper().strip()
    if cls not in list("ABCDEFG"):
        cls = "ND"
        label = "?"
    else:
        label = cls
    return f'<div class="dpe-badge dpe-{cls}">{html.escape(label)}</div>'


def copy_component(text: str):
    payload = json.dumps(text)
    components.html(
        f"""
        <div style="font-family:system-ui,-apple-system,sans-serif;display:flex;align-items:center;gap:10px;height:44px;">
          <button id="copyBtn" style="border:1px solid #cbd5e1;border-radius:10px;background:white;padding:9px 14px;cursor:pointer;font-weight:650;">Copier l'argumentaire</button>
          <span id="msg" style="font-size:13px;color:#64748b;"></span>
        </div>
        <script>
          const txt = {payload};
          document.getElementById('copyBtn').onclick = async () => {{
            try {{ await navigator.clipboard.writeText(txt); document.getElementById('msg').innerText = 'Copié'; }}
            catch(e) {{ document.getElementById('msg').innerText = 'Copie automatique bloquée par le navigateur'; }}
          }};
        </script>
        """,
        height=48,
    )


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Données immobilières publiques</div>
      <div class="hero-title">DVF Scout</div>
      <div class="hero-sub">Analyse une adresse, sélectionne les ventes réellement comparables et construit un argumentaire lisible à partir des données DVF. Le DPE public est rapproché du bien lorsqu'il peut être identifié de façon suffisamment crédible.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Search ------------------------------------
with st.container(border=True):
    st.markdown("### Bien à analyser")
    address = st.text_input(
        "Adresse",
        value="",
        placeholder="Ex. 44 rue Boursault, 75017 Paris",
        help="L'adresse est normalisée par le géocodeur public IGN/Géoplateforme.",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        target_surface = st.number_input("Surface (m²)", min_value=8.0, max_value=1500.0, value=50.0, step=0.5)
    with c2:
        target_rooms = st.number_input("Pièces principales", min_value=0, max_value=30, value=2, step=1, help="0 = ne pas privilégier le nombre de pièces")
    with c3:
        target_type = st.selectbox("Type de bien", ["Appartement", "Maison"], index=0)

    st.markdown("#### Affiner les comparables")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        radius_m = st.select_slider("Rayon", options=[150, 250, 300, 400, 500, 800, 1200], value=500, format_func=lambda x: f"{x} m")
    with r2:
        history_years = st.select_slider("Période", options=[2, 3, 4, 5], value=5, format_func=lambda x: f"{x} ans")
    with r3:
        surface_mode = st.selectbox("Tolérance surface", ["Stricte · ±15 %", "Normale · ±25 %", "Large · ±40 %"], index=1)
    with r4:
        room_mode = st.selectbox("Pièces", ["Identique", "± 1 pièce", "Ne pas filtrer"], index=1)

    with st.expander("⚙️ Paramètres avancés · mode expert", expanded=False):
        e1, e2, e3 = st.columns(3)
        with e1:
            remove_outliers = st.toggle("Écarter les valeurs aberrantes", value=True)
            include_complex = st.toggle(
                "Inclure les mutations complexes",
                value=False,
                help="À laisser désactivé dans la plupart des analyses : le prix d'une mutation comportant plusieurs biens ne peut pas toujours être affecté correctement à un seul logement.",
            )
        with e2:
            distance_emphasis = st.selectbox("Poids de la proximité", ["Faible", "Normal", "Fort"], index=1)
            recency_emphasis = st.selectbox("Poids de la récence", ["Faible", "Normal", "Fort"], index=1)
        with e3:
            building_priority = st.toggle("Prioriser le même numéro", value=True)
            max_results = st.selectbox("Nombre max. de comparables", [20, 30, 40, 50], index=1)

        qualitative = st.text_area(
            "Éléments qualitatifs du bien",
            value="",
            height=95,
            placeholder="Ex. refait à neuf, vue jardin, 1er étage, cuisine ouverte…",
            help="Ces éléments sont repris dans l'argumentaire mais ne créent pas automatiquement une prime de prix.",
        )

    run = st.button("Analyser cette adresse", type="primary", use_container_width=True)

surface_tolerance = {"Stricte · ±15 %": 0.15, "Normale · ±25 %": 0.25, "Large · ±40 %": 0.40}[surface_mode]
room_tolerance = {"Identique": 0, "± 1 pièce": 1, "Ne pas filtrer": None}[room_mode]
weight_factor = {"Faible": 0.65, "Normal": 1.0, "Fort": 1.45}

def current_signature():
    return (
        address.strip(), float(target_surface), int(target_rooms), target_type, int(radius_m), int(history_years),
        surface_mode, room_mode, bool(remove_outliers), bool(include_complex), distance_emphasis,
        recency_emphasis, bool(building_priority), int(max_results), qualitative.strip(),
    )

if run:
    if not address.strip():
        st.warning("Saisissez une adresse avant de lancer l'analyse.")
    else:
        try:
            with st.status("Analyse en cours…", expanded=True) as status:
                st.write("**1 · Adresse** — géocodage IGN/Géoplateforme")
                geo = geocode_address(address)
                st.write(f"Adresse retenue : **{geo.label}**")

                current_year = datetime.now().year
                candidate_years = tuple(range(current_year, current_year - history_years - 2, -1))
                st.write("**2 · Marché** — récupération des transactions DVF")
                raw, loaded_years = cached_load_city_dvf(geo.citycode, candidate_years, history_years)
                st.write("Millésimes chargés : " + ", ".join(map(str, loaded_years)))

                st.write("**3 · Nettoyage** — reconstruction des mutations")
                mutations = prepare_mutations(raw, target_type=target_type)
                if mutations.empty:
                    raise RuntimeError("Aucune vente exploitable n'a été trouvée pour ce type de bien.")
                mutations = enrich_location_flags(mutations, geo)

                st.write("**4 · Comparables** — sélection et scoring")
                comps = select_comparables(
                    mutations,
                    target_surface=target_surface,
                    target_rooms=int(target_rooms) if target_rooms else None,
                    radius_m=radius_m,
                    surface_tolerance=surface_tolerance,
                    room_tolerance=room_tolerance,
                    max_results=max_results,
                    include_complex=include_complex,
                    remove_outliers=remove_outliers,
                    distance_weight_factor=weight_factor[distance_emphasis],
                    recency_weight_factor=weight_factor[recency_emphasis],
                    building_priority=building_priority,
                )
                if comps.empty:
                    raise RuntimeError("Aucun comparable n'a passé les filtres. Élargissez le rayon, la tolérance de surface ou la tolérance sur les pièces.")
                estimate = estimate_market(comps, target_surface)
                summary = street_summary(mutations)
                confidence = analysis_confidence(comps, include_complex=include_complex)

                st.write("**5 · DPE** — recherche dans la base publique ADEME")
                dpe_candidates, dpe_error = cached_dpe(geo_to_tuple(geo), float(target_surface), target_type)
                best_dpe = select_best_dpe(dpe_candidates)
                if best_dpe:
                    st.write(f"DPE candidat : **{best_dpe.get('etiquette_dpe') or 'n/d'}** · rapprochement {best_dpe.get('match_confidence', 'n/d').lower()}")
                else:
                    st.write("Aucun DPE suffisamment identifiable automatiquement.")

                argument = build_argumentaire(
                    geo.label, estimate, summary, comps, qualitative,
                    dpe=best_dpe, confidence=confidence,
                )
                takeaways = build_takeaways(estimate, summary, comps, confidence, best_dpe)

                st.session_state["analysis_result"] = {
                    "signature": current_signature(),
                    "geo": geo,
                    "loaded_years": loaded_years,
                    "mutations": mutations,
                    "comps": comps,
                    "estimate": estimate,
                    "summary": summary,
                    "confidence": confidence,
                    "dpe_candidates": dpe_candidates,
                    "dpe_error": dpe_error,
                    "best_dpe": best_dpe,
                    "argument": argument,
                    "takeaways": takeaways,
                    "target_surface": target_surface,
                    "target_rooms": int(target_rooms),
                    "target_type": target_type,
                    "radius_m": radius_m,
                    "include_complex": include_complex,
                }
                status.update(label="Analyse terminée", state="complete", expanded=False)
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("analysis_result")
if not result:
    st.markdown('<div class="section-title">Ce que vous obtiendrez</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Une seule page à faire défiler, avec l’essentiel visible immédiatement et les réglages experts rangés à part.</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.info("**Marché réel**\n\nVentes DVF nettoyées, même numéro, même rue et voisinage.")
    p2.info("**DPE public**\n\nRapprochement ADEME avec niveau de confiance au lieu d'un appariement forcé.")
    p3.info("**Argumentaire**\n\nFourchette, robustesse, références et limites clairement séparées.")
    st.stop()

if result["signature"] != current_signature():
    st.warning("Vous avez modifié un ou plusieurs paramètres depuis la dernière analyse. Cliquez sur **Analyser cette adresse** pour actualiser les résultats.")

geo = result["geo"]
comps = result["comps"]
estimate = result["estimate"]
summary = result["summary"]
confidence = result["confidence"]
best_dpe = result["best_dpe"]
dpe_candidates = result["dpe_candidates"]
dpe_error = result["dpe_error"]
argument = result["argument"]
takeaways = result["takeaways"]
target_surface_r = result["target_surface"]

st.markdown(f'<div class="section-title">{html.escape(geo.label)}</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="muted-chip">' + html.escape(result["target_type"]) + '</span>'
    + f'<span class="muted-chip">{target_surface_r:.1f} m²</span>'
    + f'<span class="muted-chip">rayon {result["radius_m"]} m</span>'
    + f'<span class="muted-chip">DVF {" · ".join(map(str, result["loaded_years"]))}</span>',
    unsafe_allow_html=True,
)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Niveau central", format_eur(estimate["prix_m2_central"], True))
k2.metric("Valeur indicative", format_eur(estimate["valeur_centrale"]))
k3.metric("Fourchette observée", f"{format_eur(estimate['prix_m2_bas'], True)} → {format_eur(estimate['prix_m2_haut'], True)}")
k4.metric("Robustesse", f"{confidence['label']} · {confidence['score']}/100")

st.markdown('<div class="section-title">À retenir</div>', unsafe_allow_html=True)
for text in takeaways:
    st.markdown(f'<div class="takeaway">{html.escape(text)}</div>', unsafe_allow_html=True)

# DPE
st.markdown('<div class="section-title">DPE public</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Le rapprochement est effectué à partir de l’adresse BAN, de la surface et du type de bien. Le DPE n’est pas transformé automatiquement en prime ou décote de prix.</div>', unsafe_allow_html=True)
if best_dpe:
    d1, d2 = st.columns([1, 5])
    with d1:
        st.markdown(dpe_badge(best_dpe.get("etiquette_dpe")), unsafe_allow_html=True)
    with d2:
        dpe_date = best_dpe.get("date_etablissement_dpe")
        dpe_date_txt = dpe_date.strftime("%d/%m/%Y") if pd.notna(dpe_date) else "n/d"
        surf = best_dpe.get("surface_habitable")
        surf_txt = f"{float(surf):.1f} m²" if surf is not None and np.isfinite(float(surf)) else "n/d"
        conso = best_dpe.get("conso_5_usages_par_m2_ep")
        conso_txt = f"{float(conso):.0f} kWhEP/m²/an" if conso is not None and np.isfinite(float(conso)) else "n/d"
        ges = best_dpe.get("emission_ges_5_usages_par_m2")
        ges_txt = f"{float(ges):.0f} kgCO₂e/m²/an" if ges is not None and np.isfinite(float(ges)) else "n/d"
        st.markdown(
            f"**Classe énergie {best_dpe.get('etiquette_dpe') or 'n/d'} · GES {best_dpe.get('etiquette_ges') or 'n/d'}**  \n"
            f"DPE établi le {dpe_date_txt} · surface déclarée {surf_txt} · consommation {conso_txt} · émissions {ges_txt}  \n"
            f"**Confiance du rapprochement : {best_dpe.get('match_confidence', 'n/d')}**"
        )
        if best_dpe.get("ambiguous"):
            st.caption("Plusieurs DPE proches ont été trouvés : le candidat affiché est le mieux classé, mais le rapprochement reste ambigu.")

    with st.expander("Voir les DPE candidats retrouvés à cette adresse", expanded=False):
        ddisplay = dpe_candidates.head(10).copy()
        cols = [
            "numero_dpe", "date_etablissement_dpe", "etiquette_dpe", "etiquette_ges",
            "surface_habitable", "adresse_ban", "type_batiment", "score_match",
        ]
        ddisplay = ddisplay[[c for c in cols if c in ddisplay.columns]]
        rename = {
            "numero_dpe": "N° DPE", "date_etablissement_dpe": "Date", "etiquette_dpe": "DPE",
            "etiquette_ges": "GES", "surface_habitable": "Surface m²", "adresse_ban": "Adresse",
            "type_batiment": "Type", "score_match": "Score rapprochement",
        }
        ddisplay = ddisplay.rename(columns=rename)
        if "Date" in ddisplay.columns:
            ddisplay["Date"] = pd.to_datetime(ddisplay["Date"], errors="coerce").dt.date
        if "Surface m²" in ddisplay.columns:
            ddisplay["Surface m²"] = pd.to_numeric(ddisplay["Surface m²"], errors="coerce").round(1)
        st.dataframe(ddisplay, use_container_width=True, hide_index=True)
else:
    if dpe_error:
        st.warning("Le service DPE n'a pas répondu correctement pendant cette analyse. Les résultats DVF restent valables.")
    else:
        st.info("Aucun DPE public n'a pu être rapproché automatiquement avec assez de certitude. Cela ne signifie pas nécessairement qu'aucun DPE n'existe pour le logement.")

# Local evidence
st.markdown('<div class="section-title">Même numéro, même rue, voisinage</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">On commence par les références les plus locales avant d’élargir progressivement.</div>', unsafe_allow_html=True)
bs = summary["building"]
ss = summary["street"]
l1, l2, l3 = st.columns(3)
with l1:
    st.metric("Même numéro", f"{bs['n']} vente(s)", format_eur(bs["median"], True) if bs["n"] else "aucune médiane")
with l2:
    st.metric("Même rue", f"{ss['n']} vente(s)", format_eur(ss["median"], True) if ss["n"] else "aucune médiane")
with l3:
    st.metric("Comparables retenus", f"{len(comps)}", f"score médian {estimate['score_median']:.0f}/100")

# Market charts
st.markdown('<div class="section-title">Lecture du marché</div>', unsafe_allow_html=True)
chart_col, evol_col = st.columns(2)
with chart_col:
    fig = px.scatter(
        comps,
        x="surface_reelle_bati",
        y="prix_m2",
        size="score",
        hover_data={
            "adresse_complete": True,
            "date_mutation": "|%d/%m/%Y",
            "valeur_fonciere": ":,.0f",
            "distance_m": ":.0f",
            "score": ":.0f",
        },
        labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix au m² (€)", "score": "Score"},
        title="Comparables retenus",
    )
    fig.add_vline(x=target_surface_r, line_dash="dash", annotation_text="Bien étudié")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=390)
    st.plotly_chart(fig, use_container_width=True)

with evol_col:
    yearly = comps.assign(annee=comps["date_mutation"].dt.year).groupby("annee", as_index=False)["prix_m2"].median()
    fig2 = px.line(
        yearly, x="annee", y="prix_m2", markers=True,
        labels={"annee": "Année", "prix_m2": "Médiane €/m²"},
        title="Médiane des comparables par année",
    )
    fig2.update_layout(margin=dict(l=20, r=20, t=55, b=20), height=390)
    st.plotly_chart(fig2, use_container_width=True)

# Map
st.markdown('<div class="section-title">Carte des références</div>', unsafe_allow_html=True)
map_comps = comps.dropna(subset=["latitude", "longitude"]).copy()
if not map_comps.empty:
    map_comps["Catégorie"] = np.where(map_comps["meme_numero"], "Même numéro", np.where(map_comps["meme_rue"], "Même rue", "Comparable"))
    map_comps["Taille"] = 8 + map_comps["score"] / 12
    target_row = pd.DataFrame([{
        "latitude": geo.latitude, "longitude": geo.longitude,
        "adresse_complete": geo.label, "score": 100, "Catégorie": "Bien étudié", "Taille": 18,
        "prix_m2": np.nan, "surface_reelle_bati": target_surface_r,
    }])
    map_plot = pd.concat([map_comps, target_row], ignore_index=True, sort=False)
    figm = px.scatter_mapbox(
        map_plot,
        lat="latitude", lon="longitude", color="Catégorie", size="Taille",
        hover_name="adresse_complete",
        hover_data={"score": True, "prix_m2": ":.0f", "surface_reelle_bati": ":.1f", "Taille": False},
        zoom=14, height=470,
    )
    figm.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(figm, use_container_width=True)
else:
    st.info("Pas de coordonnées disponibles pour cartographier les références retenues.")

# Comparables table
st.markdown('<div class="section-title">Comparables classés</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Le score combine proximité, surface, récence, pièces et localisation dans la rue. Il sert à classer les références, pas à certifier une valeur.</div>', unsafe_allow_html=True)
display = comps[[
    "date_mutation", "adresse_complete", "surface_reelle_bati", "nombre_pieces_principales",
    "valeur_fonciere", "prix_m2", "distance_m", "score", "meme_numero", "meme_rue", "vente_simple",
]].copy()
display.columns = ["Date", "Adresse", "Surface m²", "Pièces", "Prix", "€/m²", "Distance m", "Score", "Même numéro", "Même rue", "Vente simple"]
display["Date"] = display["Date"].dt.date
display["Surface m²"] = display["Surface m²"].round(1)
display["Prix"] = display["Prix"].round(0).astype("Int64")
display["€/m²"] = display["€/m²"].round(0).astype("Int64")
display["Distance m"] = display["Distance m"].round(0).astype("Int64")
display["Score"] = display["Score"].round(0).astype("Int64")
st.dataframe(display, use_container_width=True, hide_index=True, height=min(650, 42 + 35 * len(display)))

# Argumentaire
st.markdown('<div class="section-title">Argumentaire factuel</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Le texte ci-dessous distingue volontairement les observations chiffrées des éléments qualitatifs qui ne peuvent pas être valorisés automatiquement.</div>', unsafe_allow_html=True)
st.text_area("Texte prêt à reprendre", value=argument, height=390, label_visibility="collapsed")
copy_component(argument)

csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
report_bytes = argument.encode("utf-8")
d1, d2 = st.columns(2)
with d1:
    st.download_button("Télécharger les comparables (.csv)", data=csv_bytes, file_name="comparables_dvf.csv", mime="text/csv", use_container_width=True)
with d2:
    st.download_button("Télécharger l'argumentaire (.txt)", data=report_bytes, file_name="argumentaire_dvf.txt", mime="text/plain", use_container_width=True)

with st.expander("Méthode, sources et limites", expanded=False):
    st.markdown(
        f"""
        **Transactions DVF.** Les mutations sont reconstruites avant calcul. Par défaut, les mutations comportant plusieurs locaux principaux sont exclues, car la valeur foncière globale ne peut pas toujours être attribuée proprement à un logement donné.

        **Comparables.** L'échantillon courant utilise un rayon de **{result['radius_m']} m** et classe les ventes selon la proximité, la similarité de surface, la récence, le nombre de pièces et la présence dans la même rue ou au même numéro.

        **Estimation.** Le niveau central est une médiane pondérée par le score. La fourchette correspond aux 25e et 75e percentiles des comparables retenus. Elle décrit l'échantillon observé ; elle n'est pas une expertise immobilière réglementée.

        **DPE.** La base ADEME contient les DPE transmis par les diagnostiqueurs depuis juillet 2021. L'appariement automatique peut être ambigu dans les copropriétés comportant plusieurs logements ou plusieurs diagnostics. C'est pourquoi DVF Scout affiche un niveau de confiance et n'utilise pas le DPE pour créer automatiquement une prime ou une décote.

        **Limites importantes.** DVF ne décrit pas précisément l'étage, la vue, l'état intérieur, la qualité des travaux, l'exposition, l'occupation ou les conditions de négociation.
        """
    )

st.caption("Sources publiques : DGFiP / data.gouv.fr · DVF géolocalisées ; IGN / Géoplateforme · géocodage ; ADEME · DPE logements existants depuis juillet 2021.")
