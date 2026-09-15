from __future__ import annotations

import io
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import requests

GEOCODE_URL = "https://data.geopf.fr/geocodage/search"
DVF_BASE = "https://files.data.gouv.fr/geo-dvf/latest/csv"


@dataclass
class GeocodeResult:
    label: str
    longitude: float
    latitude: float
    citycode: str
    postcode: str
    housenumber: str
    street: str
    city: str
    score: float | None = None


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_house_number(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    # DVF often stores numbers as floats when imported by pandas.
    try:
        f = float(text.replace(",", "."))
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return normalize_text(text)


def arrondissement_citycode(citycode: str, postcode: str) -> str:
    """Fallback for geocoders that return the parent municipality code."""
    pc = str(postcode or "")
    cc = str(citycode or "")
    if re.fullmatch(r"750(0[1-9]|1[0-9]|20)", pc):
        return "751" + pc[-2:]
    if re.fullmatch(r"130(0[1-9]|1[0-6])", pc):
        return "132" + pc[-2:]
    if re.fullmatch(r"6900[1-9]", pc):
        return "6938" + str(int(pc[-1]) - 1)
    return cc


def department_from_citycode(citycode: str) -> str:
    code = str(citycode)
    if code.startswith(("971", "972", "973", "974", "976")):
        return code[:3]
    if code.startswith("2A") or code.startswith("2B"):
        return code[:2]
    return code[:2]


def geocode_address(address: str, timeout: int = 20) -> GeocodeResult:
    params = {"q": address, "limit": 5, "autocomplete": "false"}
    r = requests.get(GEOCODE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    features = payload.get("features") or []
    if not features:
        raise ValueError("Adresse introuvable par le géocodeur IGN.")

    # Prefer a housenumber result, then the highest score.
    def key(feature):
        p = feature.get("properties", {})
        return (1 if p.get("type") == "housenumber" else 0, float(p.get("score", 0) or 0))

    feature = sorted(features, key=key, reverse=True)[0]
    props = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates") or [None, None]
    lon, lat = coords[0], coords[1]
    if lon is None or lat is None:
        raise ValueError("Le géocodeur n'a pas renvoyé de coordonnées exploitables.")

    postcode = str(props.get("postcode") or "")
    citycode = arrondissement_citycode(str(props.get("citycode") or ""), postcode)
    street = props.get("street") or props.get("name") or ""
    housenumber = props.get("housenumber") or ""

    return GeocodeResult(
        label=str(props.get("label") or address),
        longitude=float(lon),
        latitude=float(lat),
        citycode=citycode,
        postcode=postcode,
        housenumber=str(housenumber),
        street=str(street),
        city=str(props.get("city") or ""),
        score=float(props.get("score")) if props.get("score") is not None else None,
    )


def _download_one_city_year(citycode: str, year: int, timeout: int = 35) -> pd.DataFrame | None:
    dept = department_from_citycode(citycode)
    candidates = [
        f"{DVF_BASE}/{year}/communes/{dept}/{citycode}.csv",
        f"{DVF_BASE}/{year}/communes/{dept}/{citycode}.csv.gz",
    ]
    headers = {"User-Agent": "DVF-Scout-V1/1.0"}
    for url in candidates:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
        except requests.RequestException:
            continue
        if r.status_code == 404:
            continue
        r.raise_for_status()
        try:
            return pd.read_csv(io.BytesIO(r.content), low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Fichier DVF téléchargé mais illisible pour {year}: {exc}") from exc
    return None


def load_city_dvf(citycode: str, years: Iterable[int], max_loaded: int | None = None) -> tuple[pd.DataFrame, list[int]]:
    frames: list[pd.DataFrame] = []
    loaded_years: list[int] = []
    for year in years:
        df = _download_one_city_year(citycode, int(year))
        if df is not None and not df.empty:
            df = df.copy()
            df["_source_year"] = int(year)
            frames.append(df)
            loaded_years.append(int(year))
            if max_loaded is not None and len(frames) >= max_loaded:
                break
    if not frames:
        raise RuntimeError(
            "Aucun fichier DVF n'a pu être récupéré pour cette commune sur la période demandée."
        )
    return pd.concat(frames, ignore_index=True, sort=False), sorted(loaded_years)


def prepare_mutations(raw: pd.DataFrame, target_type: str = "Appartement") -> pd.DataFrame:
    df = raw.copy()
    needed = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "adresse_numero", "adresse_suffixe", "adresse_nom_voie", "code_postal",
        "code_commune", "type_local", "surface_reelle_bati",
        "nombre_pieces_principales", "latitude", "longitude", "nombre_lots",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    for col in ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "latitude", "longitude", "nombre_lots"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["nature_mutation"].astype(str).str.contains("Vente", case=False, na=False)].copy()
    if df.empty:
        return pd.DataFrame()

    principal_types = {"Appartement", "Maison"}
    rows = []

    for mutation_id, g in df.groupby("id_mutation", dropna=False, sort=False):
        gt = g[g["type_local"].astype(str).str.casefold() == target_type.casefold()].copy()
        if gt.empty:
            continue

        # Eliminate duplicated DVF lines describing the same principal local.
        dedup_cols = [
            "adresse_numero", "adresse_suffixe", "adresse_nom_voie",
            "surface_reelle_bati", "nombre_pieces_principales", "latitude", "longitude",
        ]
        target_units = gt.drop_duplicates(subset=dedup_cols)

        principal = g[g["type_local"].isin(principal_types)].drop_duplicates(subset=["type_local"] + dedup_cols)
        simple_sale = len(target_units) == 1 and len(principal) == 1

        unit = target_units.sort_values("surface_reelle_bati", ascending=False).iloc[0]
        values = g["valeur_fonciere"].dropna()
        value = float(values.iloc[0]) if not values.empty else np.nan
        surface = float(unit["surface_reelle_bati"]) if pd.notna(unit["surface_reelle_bati"]) else np.nan
        price_m2 = value / surface if value > 0 and surface > 0 else np.nan

        num = unit.get("adresse_numero")
        suffix = unit.get("adresse_suffixe")
        road = unit.get("adresse_nom_voie")
        num_txt = normalize_house_number(num)
        suffix_txt = normalize_text(suffix)
        road_txt = str(road) if pd.notna(road) else ""
        address = " ".join(x for x in [num_txt, suffix_txt, road_txt] if x).strip()

        rows.append({
            "id_mutation": mutation_id,
            "date_mutation": unit.get("date_mutation"),
            "valeur_fonciere": value,
            "surface_reelle_bati": surface,
            "nombre_pieces_principales": unit.get("nombre_pieces_principales"),
            "type_local": target_type,
            "adresse_numero": num_txt,
            "adresse_suffixe": suffix_txt,
            "adresse_nom_voie": road_txt,
            "adresse_complete": address,
            "code_postal": unit.get("code_postal"),
            "code_commune": unit.get("code_commune"),
            "latitude": unit.get("latitude"),
            "longitude": unit.get("longitude"),
            "nombre_lots": g["nombre_lots"].max(skipna=True),
            "prix_m2": price_m2,
            "vente_simple": bool(simple_sale),
            "nb_locaux_principaux": int(len(principal)),
            "nb_locaux_cible": int(len(target_units)),
            "voie_norm": normalize_text(road_txt),
            "numero_norm": num_txt,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date_mutation"] = pd.to_datetime(out["date_mutation"], errors="coerce")
    out = out.dropna(subset=["date_mutation", "valeur_fonciere", "surface_reelle_bati", "prix_m2"])
    out = out[(out["surface_reelle_bati"] > 5) & (out["prix_m2"] >= 500) & (out["prix_m2"] <= 60000)]
    return out.sort_values("date_mutation", ascending=False).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def enrich_location_flags(mutations: pd.DataFrame, geo: GeocodeResult) -> pd.DataFrame:
    df = mutations.copy()
    target_road = normalize_text(geo.street)
    target_num = normalize_house_number(geo.housenumber)

    valid_geo = df["latitude"].notna() & df["longitude"].notna()
    df["distance_m"] = np.nan
    df.loc[valid_geo, "distance_m"] = 1000 * haversine_km(
        geo.latitude,
        geo.longitude,
        df.loc[valid_geo, "latitude"].astype(float).to_numpy(),
        df.loc[valid_geo, "longitude"].astype(float).to_numpy(),
    )
    df["meme_rue"] = df["voie_norm"].eq(target_road)
    df["meme_numero"] = df["meme_rue"] & df["numero_norm"].eq(target_num) & (target_num != "")
    return df


def _apply_robust_outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 8:
        return df
    q1 = df["prix_m2"].quantile(0.25)
    q3 = df["prix_m2"].quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return df
    lower = max(500, q1 - 2.0 * iqr)
    upper = min(60000, q3 + 2.0 * iqr)
    return df[df["prix_m2"].between(lower, upper)]


def select_comparables(
    mutations: pd.DataFrame,
    target_surface: float,
    target_rooms: int | None,
    radius_m: int = 500,
    surface_tolerance: float = 0.25,
    max_results: int = 30,
) -> pd.DataFrame:
    df = mutations.copy()
    df = df[df["vente_simple"]].copy()
    df = df[df["distance_m"].notna() & (df["distance_m"] <= radius_m)]
    min_surface = max(8.0, target_surface * (1 - surface_tolerance))
    max_surface = target_surface * (1 + surface_tolerance)
    df = df[df["surface_reelle_bati"].between(min_surface, max_surface)]
    df = _apply_robust_outlier_filter(df)
    if df.empty:
        return df

    newest = df["date_mutation"].max()
    age_years = (newest - df["date_mutation"]).dt.days.clip(lower=0) / 365.25
    recency = (1 - age_years / 5).clip(lower=0, upper=1) * 20
    surf_sim = (1 - (df["surface_reelle_bati"] - target_surface).abs() / max(target_surface * 0.5, 1)).clip(0, 1) * 30
    dist_sim = (1 - df["distance_m"] / max(radius_m, 1)).clip(0, 1) * 25

    if target_rooms is not None and target_rooms > 0:
        room_diff = (df["nombre_pieces_principales"] - target_rooms).abs()
        room_score = np.select([room_diff == 0, room_diff == 1], [10, 5], default=0)
    else:
        room_score = np.full(len(df), 5.0)

    df["score"] = (
        surf_sim + dist_sim + recency + room_score
        + df["meme_rue"].astype(int) * 5
        + df["meme_numero"].astype(int) * 10
    ).round(1)
    return df.sort_values(["score", "date_mutation"], ascending=[False, False]).head(max_results).reset_index(drop=True)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = weights.sum() / 2
    idx = np.searchsorted(np.cumsum(weights), cutoff, side="left")
    return float(values[min(idx, len(values) - 1)])


def estimate_market(comps: pd.DataFrame, target_surface: float) -> dict:
    if comps.empty:
        return {}
    central = weighted_median(comps["prix_m2"].to_numpy(), np.maximum(comps["score"].to_numpy(), 1))
    q25 = float(comps["prix_m2"].quantile(0.25))
    q75 = float(comps["prix_m2"].quantile(0.75))
    median = float(comps["prix_m2"].median())
    return {
        "n": int(len(comps)),
        "prix_m2_central": central,
        "prix_m2_median": median,
        "prix_m2_bas": q25,
        "prix_m2_haut": q75,
        "valeur_centrale": central * target_surface,
        "valeur_basse": q25 * target_surface,
        "valeur_haute": q75 * target_surface,
        "score_median": float(comps["score"].median()),
        "date_plus_recente": comps["date_mutation"].max(),
        "date_plus_ancienne": comps["date_mutation"].min(),
    }


def street_summary(mutations: pd.DataFrame) -> dict:
    same_street = mutations[mutations["meme_rue"] & mutations["vente_simple"]]
    same_number = mutations[mutations["meme_numero"] & mutations["vente_simple"]]
    def stats(d):
        if d.empty:
            return {"n": 0, "median": np.nan, "latest": None}
        return {"n": len(d), "median": float(d["prix_m2"].median()), "latest": d["date_mutation"].max()}
    return {"street": stats(same_street), "building": stats(same_number)}


def format_eur(value, per_m2=False):
    if value is None or not np.isfinite(float(value)):
        return "n/d"
    txt = f"{float(value):,.0f}".replace(",", " ") + " €"
    return txt + ("/m²" if per_m2 else "")


def build_argumentaire(address: str, estimate: dict, summary: dict, comps: pd.DataFrame, qualitative: str = "") -> str:
    if not estimate:
        return "Pas assez de comparables fiables pour produire un argumentaire chiffré."
    same_b = summary["building"]
    same_s = summary["street"]
    recent = estimate["date_plus_recente"].strftime("%d/%m/%Y") if pd.notna(estimate["date_plus_recente"]) else "n/d"

    paragraphs = [
        f"Analyse DVF – {address}",
        "",
        f"Le moteur retient {estimate['n']} ventes comparables simples après nettoyage des mutations complexes et filtrage par distance, surface et valeurs aberrantes.",
        f"Le niveau central ressort à {format_eur(estimate['prix_m2_central'], True)}, avec une fourchette interquartile de {format_eur(estimate['prix_m2_bas'], True)} à {format_eur(estimate['prix_m2_haut'], True)}.",
        f"Appliqué à la surface étudiée, cela correspond à environ {format_eur(estimate['valeur_centrale'])}, dans une zone statistique de {format_eur(estimate['valeur_basse'])} à {format_eur(estimate['valeur_haute'])}.",
        f"La vente comparable la plus récente de l'échantillon date du {recent}.",
    ]
    if same_b["n"]:
        paragraphs.append(f"Même numéro dans la rue : {same_b['n']} vente(s) simple(s) observée(s), médiane {format_eur(same_b['median'], True)}.")
    else:
        paragraphs.append("Aucune vente simple exploitable n'a été retrouvée au même numéro sur la période chargée.")
    if same_s["n"]:
        paragraphs.append(f"Même rue : {same_s['n']} vente(s) simple(s), médiane {format_eur(same_s['median'], True)}.")

    top = comps.head(3)
    if not top.empty:
        paragraphs.append("")
        paragraphs.append("Trois références particulièrement comparables :")
        for _, r in top.iterrows():
            paragraphs.append(
                f"• {r['date_mutation'].strftime('%d/%m/%Y')} – {r['adresse_complete']} – "
                f"{r['surface_reelle_bati']:.1f} m² – {format_eur(r['valeur_fonciere'])} – "
                f"{format_eur(r['prix_m2'], True)} – score {r['score']:.0f}/100."
            )
    if qualitative.strip():
        paragraphs += ["", "Éléments qualitatifs renseignés : " + qualitative.strip()]
    paragraphs += [
        "",
        "Limite importante : DVF décrit la transaction mais pas l'état intérieur, l'étage, la vue, la qualité de rénovation ni les conditions commerciales. Ces éléments peuvent justifier un positionnement dans la fourchette mais ne doivent pas être transformés automatiquement en prime chiffrée sans données comparables spécifiques.",
        "Source : Demandes de valeurs foncières géolocalisées (DGFiP / data.gouv.fr).",
    ]
    return "\n".join(paragraphs)
