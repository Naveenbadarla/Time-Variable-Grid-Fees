# ============================================================
# FLEX – PV + Battery + DA + ID + AS Optimisation Dashboard
# Restored + Improved + Full Explanations
# BLOCK 1 / N
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import altair as alt
from datetime import datetime, timedelta
import hashlib

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="FLEX – PV + Battery + DA + ID + AS",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------------------------------------
# THEME SYSTEM
# ------------------------------------------------------------
FLEX_PRIMARY = "#ef4444"
FLEX_BG = "#020617"

def inject_flex_theme():
    css = f"""
    <style>
    body {{
        background: radial-gradient(circle at top left, #111827, {FLEX_BG});
        color: #e5e7eb;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }}
    .stButton>button {{
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.5);
        padding: 0.35rem 0.9rem;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        border-color: {FLEX_PRIMARY};
        color: {FLEX_PRIMARY};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def apply_theme():
    theme = st.session_state.get("theme", "dark")
    if theme == "light":
        css = """
        <style>
        body { background: #f3f4f6; color: #111827; }
        .sidebar .sidebar-content { background: #e5e7eb; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    else:
        inject_flex_theme()

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
apply_theme()

# ------------------------------------------------------------
# SIMPLE LOGIN SYSTEM (roles preserved, but no hiding content)
# ------------------------------------------------------------
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "user": hashlib.sha256("flexuser".encode()).hexdigest(),
}

def login(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🔐 <span style='color:orange;'>FLEX</span> Login</h1>", unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u, p):
            st.session_state.logged_in = True
            st.session_state.username = u
            st.success("Login successful! Loading dashboard…")
            st.rerun()
        else:
            st.error("Invalid login. Try again.")
    st.markdown("<div style='text-align:center; margin-top:40px; color:gray;'>Designed, built and conceptualised by <b style='color:orange;'>Naveen Badarla</b></div>", unsafe_allow_html=True)


            
    st.stop()

# Logged in — proceed
st.sidebar.markdown(f"👋 Welcome, **{st.session_state.username}**")

# ------------------------------------------------------------
# FULL DETAILED "QUICK EXPLANATION" (restored + polished)
# ------------------------------------------------------------
with st.sidebar.expander("💡 **Quick Explanation (Full)**", expanded=False):
    st.markdown("""
### ☀️ **1. PV → Home → Battery → Home**
- Your PV system first powers your home directly.  
- Any leftover PV that you don't immediately use can be stored in your battery.  
- Later (evening/night), the battery powers your home again.  

---

### 🔋 **2. Battery never exports to the grid**  
This model **never** lets your battery discharge into the grid.  
This keeps the model compliant with EEG rules & typical German inverter setups.

---

### 🧠 **3. Optimisation happens in two layers**
#### **Layer A — Day-Ahead (DA) Optimisation**
The model "pretends" it can charge during cheap hours and reduce imports during expensive ones.

#### **Layer B — Intraday (ID) Optimisation**
A small fraction of DA energy is fine-tuned based on even shorter-term price spreads.

You can control how aggressive this is using:
- DA spread  
- ID spread  
- Capture factors  
- ID energy factor  

---

### ⚡ **4. Ancillary Services (FCR/aFRR)**  
We model only **upward** products (battery consumes power when activated).  
This avoids export into grid and matches modern up-reserve behaviour.

You earn:
- **Capacity payments (€/MW/h)**  
You pay:
- **Activation costs (extra charging energy)**  

This is modelled with:
- Reserved FCR/aFRR kW  
- Capacity prices  
- Activation factors  
- Availability share  

---

### 🧮 **5. Final outputs you get**
- PV-only annual cost  
- Battery (non-opt) annual cost  
- DA-optimised cost  
- DA+ID improved cost  
- DA+ID+AS cost  
- Arbitrage revenues  
- Grid-import reduction  
- Sensitivity visuals  
- Live ENTSO-E pricing  

---

### 🟢 **6. Validation Engine**
A triple-checker that tells you if the results are:
- **🟢 Valid**
- **🟠 Needs attention**
- **🔴 Invalid**

It investigates:
- PV balance  
- Battery throughput  
- Arbitrage sanity  
- Cost ordering  
- AS realism  
- Market realism  

---

### 🎯 Goal  
Help you understand *how different value streams stack up*:
- Self-consumption
- DA arbitrage
- ID arbitrage
- FCR / aFRR capacity revenue
- Battery sizing impact
    """)

# ------------------------------------------------------------
# END OF BLOCK 1
# Next: BLOCK 2 adds ENTSO-E, PVGIS, market presets, AS logic, etc.
# ------------------------------------------------------------
# ============================================================
# BLOCK 2 / N
# ENTSO-E LIVE PRICES • PVGIS • MARKET PRESETS • AS LOGIC
# ============================================================

# ------------------------------------------------------------
# ENTSO-E BIDDING ZONES
# ------------------------------------------------------------
ENTSOE_ZONES = {
    "DE-LU (Germany/Luxembourg)": "10Y1001A1001A83F",
    "DE-AT-LU (historical)": "10Y1001A1001A82H",
    "AT (Austria)": "10YAT-APG----L",
    "LU (Luxembourg)": "10YLU-CEGEDEL-NQ",
}

# ------------------------------------------------------------
# ENTSO-E PRICE FETCHER
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_entsoe_da_prices(api_key: str, start: datetime, end: datetime, zone: str):
    """
    Downloads day-ahead hourly spot prices (€/MWh) → converted to €/kWh.
    """

    period_start = start.strftime("%Y%m%d%H%M")
    period_end   = end.strftime("%Y%m%d%H%M")

    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        "securityToken": api_key,
        "documentType": "A44",        # Day-ahead prices
        "in_Domain": zone,
        "out_Domain": zone,
        "periodStart": period_start,
        "periodEnd": period_end,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()

    if "<Acknowledgement_MarketDocument" in resp.text:
        raise ValueError("ENTSO-E returned acknowledgement instead of price data (rate limit?)")

    root = ET.fromstring(resp.text)

    times = []
    prices = []

    for ts in root.iter():
        if ts.tag.endswith("TimeSeries"):
            for period in ts:
                if not period.tag.endswith("Period"):
                    continue

                base_time = None

                for child in period:
                    if child.tag.endswith("timeInterval"):
                        for g in child:
                            if g.tag.endswith("start"):
                                start_iso = g.text.replace("Z", "+00:00")
                                base_time = datetime.fromisoformat(start_iso)

                if base_time is None:
                    continue

                for point in period:
                    if point.tag.endswith("Point"):
                        pos = None
                        val = None
                        for el in point:
                            if el.tag.endswith("position"):
                                pos = int(el.text)
                            if el.tag.endswith("price.amount"):
                                val = float(el.text)
                        if pos is not None and val is not None:
                            t = base_time + timedelta(hours=pos - 1)
                            times.append(t)
                            prices.append(val)

    if not times:
        raise ValueError("ENTSO-E: no price data found in XML")

    s = pd.Series(prices, index=pd.DatetimeIndex(times, tz="UTC")) / 1000.0
    return s.tz_convert("Europe/Berlin")


def derive_spreads_from_prices(price_series: pd.Series):
    """
    Computes average daily spread over time range.
    """
    if price_series.empty:
        raise ValueError("ENTSO-E series empty")

    daily = price_series.resample("1D").agg(["min", "max"])
    daily["spread"] = daily["max"] - daily["min"]

    da_spread = float(daily["spread"].mean())
    id_spread = float(da_spread * 1.5)

    return da_spread, id_spread

# ------------------------------------------------------------
# PVGIS HELPERS
# ------------------------------------------------------------
@st.cache_data(ttl=86400)
def geocode_postal_code(postal_code: str, country="Germany"):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "postalcode": postal_code,
        "country": country,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": "flex-pv-app/1.0"}

    r = requests.get(url, params=params, headers=headers, timeout=12)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise ValueError(f"No location results for postal code {postal_code}")

    return float(data[0]["lat"]), float(data[0]["lon"])


@st.cache_data(ttl=86400)
def get_pvgis_yield_for_postcode(postal_code: str):
    lat, lon = geocode_postal_code(postal_code)

    url = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
    params = {
        "lat": lat,
        "lon": lon,
        "peakpower": 1,
        "loss": 14,
        "pvtechchoice": "crystSi",
        "mountingplace": "building",
        "trackingtype": 0,
        "outputformat": "json",
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    try:
        yield_kwh_per_kwp = data["outputs"]["totals"]["fixed"]["E_y"]
    except:
        raise ValueError("Unexpected PVGIS response format")

    return float(yield_kwh_per_kwp), lat, lon

# ------------------------------------------------------------
# §14a EnWG – Module 3 DSO presets (NNE grid-fee structures)
# Prices in €/kWh, hours as 0–23
# ------------------------------------------------------------
MODULE3_PRESETS = {
    "Westnetz": {
        "nne_ht": 0.1565,
        "nne_st": 0.0953,
        "nne_nt": 0.0095,
        "ht_hours": [15,16,17,18,19,20],
        "nt_hours": [0,1,2,3,4,5,6,13,14,21,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "Avacon": {
        "nne_ht": 0.0841,
        "nne_st": 0.0604,
        "nne_nt": 0.0060,
        "ht_hours": [16,17,18,19],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "MVV Netze": {
        "nne_ht": 0.0596,
        "nne_st": 0.0432,
        "nne_nt": 0.0173,
        "ht_hours": [17,18,19],
        "nt_hours": [0,1,2,3,4,5,6,23],
        "valid_quarters": [1,2,3,4],
    },
    "MITNETZ": {
        "nne_ht": 0.1260,
        "nne_st": 0.0631,
        "nne_nt": 0.0069,
        "ht_hours": [16,17,18,19,20,21],
        "nt_hours": [0,1,2,3,4,5,6,14,15,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "Stadtwerke München": {
        "nne_ht": 0.0714,
        "nne_st": 0.0647,
        "nne_nt": 0.0259,
        "ht_hours": [11,12,13,17,18],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "Thüringen Energie": {
        "nne_ht": 0.0862,
        "nne_st": 0.0556,
        "nne_nt": 0.0167,
        "ht_hours": [16,17,18,19],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "LEW": {
        "nne_ht": 0.0809,
        "nne_st": 0.0409,
        "nne_nt": 0.0041,
        "ht_hours": [17,18,19,20],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "Netze BW": {
        "nne_ht": 0.1320,
        "nne_st": 0.0757,
        "nne_nt": 0.0303,
        "ht_hours": [17,18,19,20],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "Bayernwerk": {
        "nne_ht": 0.0903,
        "nne_st": 0.0472,
        "nne_nt": 0.0047,
        "ht_hours": [17,18,19,20],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
    "EAM Netz": {
        "nne_ht": 0.1052,
        "nne_st": 0.0548,
        "nne_nt": 0.0164,
        "ht_hours": [16,17,18,19],
        "nt_hours": [0,1,2,3,4,5,6,22,23],
        "valid_quarters": [1,2,3,4],
    },
}

# ------------------------------------------------------------
# MARKET PRESETS
# ------------------------------------------------------------
def get_market_presets():
    """
    Pre-built realistic German market states.
    """
    return {
        "Conservative": {
            "da_spread": 0.07,
            "opt_cap": 0.60,
            "nonopt_cap": 0.25,
            "id_spread": 0.10,
            "id_cap": 0.50,
            "id_energy_factor": 0.12,
        },
        "Base Case (2024-2025-ish)": {
            "da_spread": 0.112,
            "opt_cap": 0.70,
            "nonopt_cap": 0.35,
            "id_spread": 0.18,
            "id_cap": 0.60,
            "id_energy_factor": 0.15,
        },
        "High Volatility": {
            "da_spread": 0.18,
            "opt_cap": 0.80,
            "nonopt_cap": 0.40,
            "id_spread": 0.30,
            "id_cap": 0.70,
            "id_energy_factor": 0.22,
        },
    }


# ------------------------------------------------------------
# AS DEFAULTS + REALISM RULES
# ------------------------------------------------------------
def auto_suggest_as(st_state):
    """
    Populates realistic AS defaults based on inverter limit.
    """
    lim = st_state["inverter_limit_kw"]

    st_state["fcr_power_kw"] = min(4.0, lim * 0.8)
    st_state["afrr_power_kw"] = min(1.0, lim * 0.4)

    st_state["fcr_price_eur_per_mw_h"] = 12.0
    st_state["afrr_price_eur_per_mw_h"] = 10.0

    st_state["fcr_activation_factor"] = 0.05
    st_state["afrr_activation_factor"] = 0.08
    st_state["as_availability_pct"]   = 80

    return st_state


def warn_as_realism(batt_kwh, inv_kw, fcr_kw, afrr_kw,
                    fcr_price, afrr_price,
                    fcr_act, afrr_act,
                    availability):
    """
    Produces instantaneous warning messages in sidebar.
    """
    if fcr_kw > inv_kw:
        st.sidebar.warning("⚠️ FCR kW exceeds inverter AC limit.")
    if afrr_kw > inv_kw:
        st.sidebar.warning("⚠️ aFRR kW exceeds inverter AC limit.")

    if batt_kwh > 0:
        if fcr_kw > batt_kwh / 2:
            st.sidebar.warning("⚠️ FCR kW unusually high vs battery size.")
        if afrr_kw > batt_kwh / 3:
            st.sidebar.warning("⚠️ aFRR kW unusually high vs battery size.")

    if fcr_price > 30:
        st.sidebar.warning("⚠️ FCR price > 30 €/MW/h — very unlikely.")
    if afrr_price > 25:
        st.sidebar.warning("⚠️ aFRR price > 25 €/MW/h — unrealistic.")

    if fcr_act > 0.12:
        st.sidebar.warning("⚠️ FCR activation factor > 12% cycles.")
    if afrr_act > 0.15:
        st.sidebar.warning("⚠️ aFRR activation factor > 15% cycles.")

    if availability > 0.95:
        st.sidebar.warning("⚠️ AS availability > 95% — unrealistic.")


# ------------------------------------------------------------
# END OF BLOCK 2
# Next block: BLOCK 3 (Core Scenario Engine & Validation Engine)
# ------------------------------------------------------------
# ============================================================
# BLOCK 3 / N
# CORE SCENARIO ENGINE + VALIDATION ENGINE
# ============================================================

# ------------------------------------------------------------
# SCENARIO ENGINE (PV-only → Battery → DA → DA+ID → AS)
# ------------------------------------------------------------
def compute_scenario(
    load_kwh: float,
    pv_kwp: float,
    pv_yield: float,
    grid_price: float,
    fit_price: float,
    batt_capacity: float,
    batt_eff: float,
    cycles_per_day: float,
    sc_ratio_no_batt: float,
    da_spread: float,
    opt_capture: float,
    nonopt_capture: float,
    id_spread: float,
    id_capture: float,
    id_energy_factor: float,
    max_throughput_factor: float,
    as_enabled: bool,
    fcr_power_kw: float,
    fcr_price_eur_per_mw_h: float,
    fcr_activation_factor: float,
    afrr_power_kw: float,
    afrr_price_eur_per_mw_h: float,
    afrr_activation_factor: float,
    as_availability_share: float,
):
    """
    Main FLEX engine. Computes ALL configurations.
    Battery is NEVER allowed to export to grid (EEG compliant).
    """

    # --- PV production ---
    pv_gen = pv_kwp * pv_yield  # kWh/year

    # --------------------------------------------------------
    # 1. PV-only (no battery)
    # --------------------------------------------------------
    pv_direct_sc = min(load_kwh * sc_ratio_no_batt, pv_gen)
    pv_export_no_batt = max(0.0, pv_gen - pv_direct_sc)
    grid_import_no_batt = max(0.0, load_kwh - pv_direct_sc)

    cost_no_batt = grid_import_no_batt * grid_price
    revenue_no_batt = pv_export_no_batt * fit_price
    net_no_batt = cost_no_batt - revenue_no_batt

    # --------------------------------------------------------
    # 2. Battery – NON-OPTIMISED
    # --------------------------------------------------------
    theoretical_throughput = (
        batt_capacity * batt_eff * cycles_per_day * 365.0
    ) * max_throughput_factor

    remaining_load = max(0.0, load_kwh - pv_direct_sc)
    batt_usable = min(remaining_load, theoretical_throughput)

    pv_to_batt = batt_usable / batt_eff if batt_eff > 0 else 0.0
    pv_export_batt = max(0.0, pv_gen - pv_direct_sc - pv_to_batt)

    grid_import_batt = max(0.0, load_kwh - (pv_direct_sc + batt_usable))

    cost_batt_base = grid_import_batt * grid_price
    revenue_batt = pv_export_batt * fit_price
    net_batt_base = cost_batt_base - revenue_batt

    # Arbitrage energy = battery discharge energy
    arbitrage_energy = batt_usable if load_kwh > 0 else 0.0

    arbitrage_non = arbitrage_energy * da_spread * nonopt_capture
    net_batt_nonopt = net_batt_base - arbitrage_non

    # --------------------------------------------------------
    # 3. Battery – DA-optimised
    # --------------------------------------------------------
    arbitrage_opt = arbitrage_energy * da_spread * opt_capture
    net_batt_opt = net_batt_base - arbitrage_opt

    # --------------------------------------------------------
    # 4. Battery – DA+ID-optimised
    # --------------------------------------------------------
    id_energy = arbitrage_energy * max(0.0, min(id_energy_factor, 1.0))
    id_arbitrage = id_energy * id_spread * id_capture

    net_batt_da_id = net_batt_base - (arbitrage_opt + id_arbitrage)

    # --------------------------------------------------------
    # Assemble results (WITHOUT AS)
    # --------------------------------------------------------
    rows = [
        {
            "Configuration": "PV-only (No battery)",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": 0.0,
            "PV export (kWh)": pv_export_no_batt,
            "Grid import (kWh)": grid_import_no_batt,
            "Grid cost (€)": cost_no_batt,
            "EEG revenue (€)": revenue_no_batt,
            "DA arbitrage (€)": 0.0,
            "ID arbitrage (€)": 0.0,
            "Net annual cost (€)": net_no_batt,
        },
        {
            "Configuration": "Battery – non-optimised",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_non,
            "ID arbitrage (€)": 0.0,
            "Net annual cost (€)": net_batt_nonopt,
        },
        {
            "Configuration": "Battery – DA-optimised",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_opt,
            "ID arbitrage (€)": 0.0,
            "Net annual cost (€)": net_batt_opt,
        },
        {
            "Configuration": "Battery – DA+ID-optimised",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_opt,
            "ID arbitrage (€)": id_arbitrage,
            "Net annual cost (€)": net_batt_da_id,
        },
    ]

    # --------------------------------------------------------
    # 5. Battery – DA+ID + Ancillary Services (Up-only)
    # --------------------------------------------------------
    if as_enabled and (fcr_power_kw > 0 or afrr_power_kw > 0):
        hours = 8760 * as_availability_share

        fcr_mw = fcr_power_kw / 1000
        afrr_mw = afrr_power_kw / 1000

        fcr_capacity_rev  = fcr_mw  * fcr_price_eur_per_mw_h  * hours
        afrr_capacity_rev = afrr_mw * afrr_price_eur_per_mw_h * hours

        theoretical_energy = batt_capacity * batt_eff * cycles_per_day * 365
        total_activation_energy = theoretical_energy * (
            fcr_activation_factor + afrr_activation_factor
        ) * as_availability_share

        total_activation_energy = max(0.0, total_activation_energy)
        activation_cost = total_activation_energy * grid_price

        net_with_as = (
            net_batt_da_id - fcr_capacity_rev - afrr_capacity_rev + activation_cost
        )

        rows.append({
            "Configuration": "Battery – DA+ID + AS (up-only)",
            "PV generation (kWh)": pv_gen,
            "PV self-consumption (kWh)": pv_direct_sc,
            "Battery → load (kWh)": batt_usable,
            "PV export (kWh)": pv_export_batt,
            "Grid import (kWh)": grid_import_batt,
            "Grid cost (€)": cost_batt_base + activation_cost,
            "EEG revenue (€)": revenue_batt,
            "DA arbitrage (€)": arbitrage_opt,
            "ID arbitrage (€)": id_arbitrage,
            "FCR capacity revenue (€)": fcr_capacity_rev,
            "aFRR capacity revenue (€)": afrr_capacity_rev,
            "AS activation energy (kWh)": total_activation_energy,
            "AS activation cost (€)": activation_cost,
            "Net annual cost (€)": net_with_as,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# VALIDATION ENGINE (🟢 / 🟠 / 🔴)
# ------------------------------------------------------------
def validate_results(
    df: pd.DataFrame,
    load_kwh: float,
    pv_kwp: float,
    pv_yield: float,
    batt_capacity: float,
    batt_eff: float,
    cycles: float,
    max_throughput_factor: float,
    da_spread: float,
    id_spread: float,
    id_energy_factor: float,
    as_enabled: bool,
    fcr_activation_factor: float,
    afrr_activation_factor: float,
):
    """
    Deep validation of physics, economics, AS realism.
    """
    messages = []
    def add(level, msg):
        messages.append((level, msg))

    configs = df.set_index("Configuration")

    # ------------------------------------------
    # 1) Required configurations present?
    # ------------------------------------------
    required = [
        "PV-only (No battery)",
        "Battery – non-optimised",
        "Battery – DA-optimised",
        "Battery – DA+ID-optimised",
    ]
    missing = [c for c in required if c not in configs.index]
    if missing:
        add("red", f"Missing key configurations: {missing}")
        return "red", messages

    # ------------------------------------------
    # 2) PV balance
    # ------------------------------------------
    try:
        r = configs.loc["Battery – DA+ID-optimised"]
        pv_gen = r["PV generation (kWh)"]
        pv_sc = r["PV self-consumption (kWh)"]
        batt_load = r["Battery → load (kWh)"]
        pv_export = r["PV export (kWh)"]

        pv_to_batt = batt_load / batt_eff if batt_eff > 0 else 0
        balance = pv_gen - (pv_sc + pv_export + pv_to_batt)
        tol = max(0.1, pv_gen * 0.01)

        if abs(balance) <= tol:
            add("green", "PV balance ✔ (generation = SC + export + battery charge)")
        elif abs(balance) <= pv_gen * 0.03:
            add("orange", "Minor PV imbalance (≈3%). Usually OK.")
        else:
            add("red", "Major PV imbalance — results unreliable.")
    except Exception:
        add("red", "PV balance check failed (missing values)")

    # ------------------------------------------
    # 3) Battery throughput realism
    # ------------------------------------------
    theoretical = batt_capacity * batt_eff * cycles * 365 * max_throughput_factor
    actual = configs.loc["Battery – DA+ID-optimised"]["Battery → load (kWh)"]

    if actual <= theoretical * 1.02:
        add("green", "Battery throughput within realistic range.")
    elif actual <= theoretical * 1.2:
        add("orange", "Battery throughput slightly too high (check cycles / throughput limit).")
    else:
        add("red", "Battery throughput unrealistically high!")

    # ------------------------------------------
    # 4) Arbitrage sanity check
    # ------------------------------------------
    da_opt = configs.loc["Battery – DA-optimised"]["DA arbitrage (€)"]
    da_non = configs.loc["Battery – non-optimised"]["DA arbitrage (€)"]
    if da_opt < da_non:
        add("orange", "DA optimisation < non-optimised — check DA parameters.")

    # ID arbitrage too high?
    da_id = configs.loc["Battery – DA+ID-optimised"]["DA arbitrage (€)"]
    id_id = configs.loc["Battery – DA+ID-optimised"]["ID arbitrage (€)"]
    if da_id > 0 and id_id > da_id * 1.5:
        add("orange", "ID arbitrage unusually large vs DA.")

    # ------------------------------------------
    # 5) Cost monotonicity
    # ------------------------------------------
    try:
        c_pv  = configs.loc["PV-only (No battery)"]["Net annual cost (€)"]
        c_n   = configs.loc["Battery – non-optimised"]["Net annual cost (€)"]
        c_d   = configs.loc["Battery – DA-optimised"]["Net annual cost (€)"]
        c_di  = configs.loc["Battery – DA+ID-optimised"]["Net annual cost (€)"]

        if c_pv < c_n: add("orange", "PV-only cheaper than non-optimised battery? Check inputs.")
        if c_n < c_d:  add("orange", "Non-optimised cheaper than DA-optimised? Check spreads.")
        if c_d < c_di: add("orange", "DA-optimised cheaper than DA+ID? Check ID parameters.")
    except:
        add("orange", "Cost ordering check incomplete.")

    # ------------------------------------------
    # 6) AS realism
    # ------------------------------------------
    if as_enabled and "Battery – DA+ID + AS (up-only)" in configs.index:
        r_as = configs.loc["Battery – DA+ID + AS (up-only)"]
        as_energy = r_as.get("AS activation energy (kWh)", 0) or 0
        share = (
            as_energy /
            (batt_capacity * batt_eff * cycles * 365 * max_throughput_factor)
            if batt_capacity > 0 else 0
        )

        if share <= 0.2:
            add("green", "AS activation energy in realistic range.")
        elif share <= 0.5:
            add("orange", "AS activation share quite high—borderline.")
        else:
            add("red", "AS activation energy unrealistically high.")

        if fcr_activation_factor > 0.12:
            add("orange", "FCR activation factor >12% cycles (unrealistic).")
        if afrr_activation_factor > 0.15:
            add("orange", "aFRR activation >15% cycles (unrealistic).")

    # ------------------------------------------
    # 7) Market realism
    # ------------------------------------------
    if da_spread > 0.30:
        add("orange", "DA spread unusually high (>0.30 €/kWh).")
    if id_spread > 0.60:
        add("orange", "ID spread extremely high (>0.60 €/kWh).")
    if id_energy_factor > 0.30:
        add("orange", "ID energy factor >0.30 (very optimistic).")

    # ------------------------------------------
    # FINAL STATUS
    # ------------------------------------------
    levels = [lvl for lvl, _ in messages]
    if "red" in levels:
        return "red", messages
    if "orange" in levels:
        return "orange", messages
    return "green", messages


# ------------------------------------------------------------
# END OF BLOCK 3
# NEXT: BLOCK 4 = FULL MAIN APP (Sidebar Inputs + AS UI)
# ------------------------------------------------------------
# ============================================================
# BLOCK 4 / N
# SIDEBAR UI — PV • BATTERY • MARKET • PVGIS • ENTSO-E • AS
# ============================================================

def build_sidebar_inputs():
    st.sidebar.header("🔧 System Setup")

    # Dark / Light toggle
    dark_on = st.sidebar.toggle(
        "🌗 Dark mode",
        value=(st.session_state.get("theme", "dark") == "dark")
    )
    st.session_state["theme"] = "dark" if dark_on else "light"
    apply_theme()

    # -------------- BASIC SYSTEM INPUTS --------------
    load_kwh = st.sidebar.number_input(
        "Annual household load (kWh)",
        min_value=0.0, value=3000.0, step=500.0,
        help="Annual electricity consumption."
    )
    pv_kwp = st.sidebar.number_input(
        "PV size (kWp)",
        min_value=0.0, max_value=40.0, value=9.5, step=0.1
    )

    # -------------- PVGIS yield --------------
    postal_code = st.sidebar.text_input(
        "Postal code (Germany) – for PVGIS",
        help="Enter a German postal code to fetch realistic PV yield."
    )

    if st.sidebar.button("☀️ Fetch PV yield from PVGIS"):
        if not postal_code.strip():
            st.sidebar.error("Please enter a postal code first.")
        else:
            try:
                with st.spinner("🔍 Contacting PVGIS…"):
                    est_yield, lat, lon = get_pvgis_yield_for_postcode(postal_code)
                st.session_state["pvgis_yield"] = round(est_yield, 1)
                st.sidebar.success(f"PVGIS: ~{est_yield:.0f} kWh/kWp/year")
            except Exception as e:
                st.sidebar.error(f"PVGIS error: {e}")

    default_yield = st.session_state.get("pvgis_yield") or 950.0
    pv_yield = st.sidebar.number_input(
    "PV yield (kWh/kWp·yr)",
    min_value=600.0,
    max_value=1500.0,
    value=float(default_yield),
    step=10.0
)

    # -------------- Prices --------------
    grid_price = st.sidebar.number_input(
        "Grid price (€/kWh)",
        min_value=0.0, max_value=1.0,
        value=0.39, step=0.01
    )
    fit_price = st.sidebar.number_input(
        "Feed-in tariff (€/kWh)",
        min_value=0.0, max_value=1.0,
        value=0.08, step=0.01
    )

    # -------------- Battery --------------
    batt_capacity = st.sidebar.number_input(
        "Battery capacity (kWh)",
        min_value=0.0, max_value=60.0,
        value=8.8, step=0.1
    )
    batt_eff = st.sidebar.slider(
        "Battery roundtrip efficiency (%)",
        min_value=60, max_value=100, value=93
    ) / 100

    cycles = st.sidebar.number_input(
        "Cycle count per day",
        min_value=0.0, max_value=5.0,
        value=1.0, step=0.1
    )

    max_throughput_factor = st.sidebar.slider(
        "Battery throughput limit (% of theoretical)",
        min_value=10, max_value=100, value=100, step=5
    ) / 100

    sc_ratio = st.sidebar.slider(
        "Self-consumption ratio (no battery)",
        min_value=0.0, max_value=1.0,
        value=0.80, step=0.05
    )

    # ============================================================
    # MARKET PARAMETERS
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Market Parameters")

    mode = st.sidebar.radio(
        "Market mode",
        ["Presets", "Manual (expert)"]
    )

    presets = get_market_presets()

    if mode == "Presets":
        chosen = st.sidebar.selectbox(
            "Preset",
            list(presets.keys()),
            index=1
        )
        p = presets[chosen]

        da_spread  = p["da_spread"]
        opt_cap    = p["opt_cap"]
        nonopt_cap = p["nonopt_cap"]
        id_spread  = p["id_spread"]
        id_cap     = p["id_cap"]
        id_energy_factor = p["id_energy_factor"]

        st.sidebar.info(
            f"DA spread: {da_spread:.3f} €/kWh\n"
            f"ID spread: {id_spread:.3f} €/kWh"
        )

    else:
        da_spread = st.sidebar.number_input(
            "DA spread (€/kWh)", 0.0, 0.5, 0.112, 0.01
        )
        opt_cap = st.sidebar.slider(
            "DA optimiser capture", 0.0, 1.0, 0.7
        )
        nonopt_cap = st.sidebar.slider(
            "Non-optimised DA capture", 0.0, 1.0, 0.35
        )
        id_spread = st.sidebar.number_input(
            "ID spread (€/kWh)", 0.0, 1.0, 0.18, 0.01
        )
        id_cap = st.sidebar.slider(
            "ID capture", 0.0, 1.0, 0.6
        )
        id_energy_factor = st.sidebar.slider(
            "Fraction of DA energy used also for ID",
            0.0, 0.7, 0.15, 0.01
        )

    # ============================================================
    # ENTSO-E Live Market Input
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Live Market (ENTSO-E)")

    use_entsoe = st.sidebar.checkbox(
        "Use ENTSO-E live prices",
        value=False
    )

    zone_name = st.sidebar.selectbox(
        "Bidding zone",
        list(ENTSOE_ZONES.keys()),
        index=0
    )
    zone_code = ENTSOE_ZONES[zone_name]

    if use_entsoe:
        api_key = st.secrets.get("ENTSOE_API_KEY")
        if not api_key:
            st.sidebar.error("Missing ENTSOE_API_KEY in secrets.")
        else:
            try:
                end_dt   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                start_dt = end_dt - timedelta(days=10)

                with st.spinner("📡 Fetching ENTSO-E data…"):
                    series = fetch_entsoe_da_prices(api_key, start_dt, end_dt, zone_code)

                da_live, id_live = derive_spreads_from_prices(series)
                da_spread = da_live
                id_spread = id_live

                st.session_state["entsoe_prices"] = series
                st.sidebar.success("Live DA/ID spreads loaded from ENTSO-E!")

            except Exception as e:
                st.sidebar.error(f"ENTSO-E fetch error: {e}")
                st.session_state["entsoe_prices"] = None

    # ============================================================
    # ANCILLARY SERVICES (AS) UI
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛰 Ancillary Services (Up-only)")

    as_enabled = st.sidebar.checkbox(
        "Enable FCR/aFRR",
        value=False,
        help="Models only upward reserve (charging-only activation). No battery export."
    )

    ss = st.session_state

    # Initialise defaults if first load
    defaults = {
        "inverter_limit_kw": 5.0,
        "as_availability_pct": 80,
        "fcr_power_kw": 4.0,
        "afrr_power_kw": 1.0,
        "fcr_price_eur_per_mw_h": 12.0,
        "afrr_price_eur_per_mw_h": 10.0,
        "fcr_activation_factor": 0.05,
        "afrr_activation_factor": 0.08,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

    if as_enabled:
        inverter_limit_kw = st.sidebar.number_input(
            "Inverter AC limit (kW)",
            1.0, 30.0,
            value=ss["inverter_limit_kw"],
            key="inverter_limit_kw"
        )

        if st.sidebar.button("✨ Auto-suggest AS parameters"):
            auto_suggest_as(ss)
            st.sidebar.success("Realistic default AS parameters applied!")

        availability_pct = st.sidebar.slider(
            "Availability (%)",
            10, 100,
            value=ss["as_availability_pct"],
            key="as_availability_pct"
        )
        as_availability_share = availability_pct / 100

        # FCR
        st.sidebar.markdown("### ⚡ FCR (Up-only)")
        fcr_power_kw = st.sidebar.number_input(
            "Reserved FCR power (kW)",
            min_value=0.0, max_value=2000.0,
            value=ss["fcr_power_kw"],
            step=0.5, key="fcr_power_kw"
        )
        fcr_price_eur_per_mw_h = st.sidebar.number_input(
            "FCR price (€/MW/h)",
            0.0, 500.0,
            value=ss["fcr_price_eur_per_mw_h"],
            step=0.1, key="fcr_price_eur_per_mw_h"
        )
        fcr_activation_factor = st.sidebar.slider(
            "FCR activation factor",
            0.0, 0.3,
            value=ss["fcr_activation_factor"],
            step=0.01, key="fcr_activation_factor"
        )

        # aFRR
        st.sidebar.markdown("### ⚡ aFRR (Up-only)")
        afrr_power_kw = st.sidebar.number_input(
            "Reserved aFRR power (kW)",
            min_value=0.0, max_value=2000.0,
            value=ss["afrr_power_kw"],
            step=0.5, key="afrr_power_kw"
        )
        afrr_price_eur_per_mw_h = st.sidebar.number_input(
            "aFRR price (€/MW/h)",
            0.0, 500.0,
            value=ss["afrr_price_eur_per_mw_h"],
            step=0.1, key="afrr_price_eur_per_mw_h"
        )
        afrr_activation_factor = st.sidebar.slider(
            "aFRR activation factor",
            0.0, 0.5,
            value=ss["afrr_activation_factor"],
            step=0.01, key="afrr_activation_factor"
        )

        # Realism warnings
        warn_as_realism(
            batt_kwh=batt_capacity,
            inv_kw=inverter_limit_kw,
            fcr_kw=fcr_power_kw,
            afrr_kw=afrr_power_kw,
            fcr_price=fcr_price_eur_per_mw_h,
            afrr_price=afrr_price_eur_per_mw_h,
            fcr_act=fcr_activation_factor,
            afrr_act=afrr_activation_factor,
            availability=as_availability_share
        )

    else:
        # If AS disabled, all AS values = 0
        fcr_power_kw = 0
        afrr_power_kw = 0
        fcr_price_eur_per_mw_h = 0
        afrr_price_eur_per_mw_h = 0
        fcr_activation_factor = 0
        afrr_activation_factor = 0
        as_availability_share = 0

    # Return everything needed
    return (
        load_kwh, pv_kwp, pv_yield, grid_price, fit_price,
        batt_capacity, batt_eff, cycles, max_throughput_factor, sc_ratio,
        da_spread, opt_cap, nonopt_cap, id_spread, id_cap, id_energy_factor,
        as_enabled, fcr_power_kw, fcr_price_eur_per_mw_h, fcr_activation_factor,
        afrr_power_kw, afrr_price_eur_per_mw_h, afrr_activation_factor,
        as_availability_share
    )

# ------------------------------------------------------------
# END OF BLOCK 4
# NEXT: BLOCK 5 (Main App: scenario run, validation banner, all tabs)
# ------------------------------------------------------------
# ============================================================
# BLOCK 5 / N
# MAIN APP — RUN ENGINE • VALIDATION BANNER • TABS STRUCTURE
# ============================================================

# ------------------------------------------------------------
# Run sidebar & scenario
# ------------------------------------------------------------
(
    load_kwh, pv_kwp, pv_yield, grid_price, fit_price,
    batt_capacity, batt_eff, cycles, max_throughput_factor, sc_ratio,
    da_spread, opt_cap, nonopt_cap, id_spread, id_cap, id_energy_factor,
    as_enabled, fcr_power_kw, fcr_price_eur_per_mw_h, fcr_activation_factor,
    afrr_power_kw, afrr_price_eur_per_mw_h, afrr_activation_factor,
    as_availability_share
) = build_sidebar_inputs()

# ------------------------------------------------------------
# RUN SCENARIO ENGINE
# ------------------------------------------------------------
df = compute_scenario(
    load_kwh=load_kwh,
    pv_kwp=pv_kwp,
    pv_yield=pv_yield,
    grid_price=grid_price,
    fit_price=fit_price,
    batt_capacity=batt_capacity,
    batt_eff=batt_eff,
    cycles_per_day=cycles,
    sc_ratio_no_batt=sc_ratio,
    da_spread=da_spread,
    opt_capture=opt_cap,
    nonopt_capture=nonopt_cap,
    id_spread=id_spread,
    id_capture=id_cap,
    id_energy_factor=id_energy_factor,
    max_throughput_factor=max_throughput_factor,
    as_enabled=as_enabled,
    fcr_power_kw=fcr_power_kw,
    fcr_price_eur_per_mw_h=fcr_price_eur_per_mw_h,
    fcr_activation_factor=fcr_activation_factor,
    afrr_power_kw=afrr_power_kw,
    afrr_price_eur_per_mw_h=afrr_price_eur_per_mw_h,
    afrr_activation_factor=afrr_activation_factor,
    as_availability_share=as_availability_share,
)

# ------------------------------------------------------------
# VALIDATION ENGINE
# ------------------------------------------------------------
status, v_msgs = validate_results(
    df=df,
    load_kwh=load_kwh,
    pv_kwp=pv_kwp,
    pv_yield=pv_yield,
    batt_capacity=batt_capacity,
    batt_eff=batt_eff,
    cycles=cycles,
    max_throughput_factor=max_throughput_factor,
    da_spread=da_spread,
    id_spread=id_spread,
    id_energy_factor=id_energy_factor,
    as_enabled=as_enabled,
    fcr_activation_factor=fcr_activation_factor,
    afrr_activation_factor=afrr_activation_factor,
)

# ------------------------------------------------------------
# VALIDATION BANNER (🟢 🟠 🔴)
# ------------------------------------------------------------
if status == "green":
    banner_color = "#065f46"
    banner_border = "#10b981"
    banner_icon = "🟢"
    banner_title = "Optimisation Valid"
    banner_text = "All core checks passed. Results look physically consistent & realistic."
elif status == "orange":
    banner_color = "#78350f"
    banner_border = "#fbbf24"
    banner_icon = "🟠"
    banner_title = "Optimisation Needs Attention"
    banner_text = "Some checks raised warnings. Results are still usable but interpret carefully."
else:
    banner_color = "#7f1d1d"
    banner_border = "#f87171"
    banner_icon = "🔴"
    banner_title = "Optimisation Invalid"
    banner_text = "At least one check indicates an inconsistency. Results may be unreliable."

st.markdown(
    f"""
    <div style="
        border: 1px solid {banner_border};
        background-color: {banner_color};
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        color: #f9fafb;
    ">
        <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 0.25rem;">
            {banner_icon} {banner_title}
        </div>
        <div style="font-size: 0.9rem;">
            {banner_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🔍 Validation details"):
    em = {"green": "✅", "orange": "⚠️", "red": "❌"}
    for lvl, msg in v_msgs:
        st.markdown(f"- {em.get(lvl,'•')} {msg}")
# ------------------------------------------------------------
# FLEX SUMMARY FOR COMPETITOR BENCHMARKING
# ------------------------------------------------------------
best_config = df.loc[df["Net annual cost (€)"].idxmin()]

# Simple baseline = all energy bought from grid
baseline_cost = load_kwh * grid_price
my_best_cost = float(best_config["Net annual cost (€)"])
my_best_savings = baseline_cost - my_best_cost

# Use your input sizes for FLEX system
flex_pv_kwp = float(pv_kwp)
flex_batt_kwh = float(batt_capacity)

# ============================================================
# TABS STRUCTURE (CONTENT WILL COME IN BLOCKS 6+)
# ============================================================

tabs = st.tabs([
    "📊 Results",
    "🧠 Optimisation Logic",
    "📘 How to understand results",
    "📚 Model Assumptions",
    "⚠️ Limitations",
    "🛰 Ancillary Services Explained",
    "🧪 Sensitivity",
    "📡 Live Market",
    "👥 Multi-audience Summary",
    "Competitor Benchmark",
    "Competitor Alerts",
    "🛠 Admin Panel"
])

  
# ------------------------------------------------------------
# END OF BLOCK 5
# NEXT BLOCK = BLOCK 6
# (Full Results KPIs + Charts + Value Stack Visuals)
# ------------------------------------------------------------
# ============================================================
# BLOCK 6 / N
# RESULTS TAB — KPIs • COST BARS • VALUE STACK • SAVINGS
# ============================================================

# ============================================================
# BLOCK 6 (REPLACED) — SIMPLE + EDUCATIONAL RESULTS TAB
# ============================================================

with tabs[0]:

    st.header("📊 Results Overview (Simple & Easy to Understand)")

    # ============================================================
    # 1) TOP-LEVEL KPIs
    # ============================================================

    c_grid = load_kwh * grid_price
    c_best = df["Net annual cost (€)"].min()
    best_config = df.loc[df["Net annual cost (€)"].idxmin(), "Configuration"]

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Grid-only cost", f"{c_grid:,.0f} €")

    with colB:
        st.metric("Best scenario cost", f"{c_best:,.0f} €")

    with colC:
        st.metric(
            "Savings vs Grid",
            f"{c_grid - c_best:,.0f} €",
            delta=f"{((c_grid - c_best)/c_grid)*100:.1f}%"
        )

    

    # ============================================================
    # SAFETY: Ensure AS columns exist
    # ============================================================
    for col in ["FCR capacity revenue (€)", "aFRR capacity revenue (€)", "AS activation cost (€)"]:
        if col not in df.columns:
            df[col] = 0.0

    # ============================================================
    # CALCULATE VALUE COMPONENTS (ALWAYS POSITIVE FOR PIE CHART)
    # ============================================================

    # We convert everything to *positive* absolute value for readability.
    value_pv = abs(df.loc[df["Configuration"] == best_config, "EEG revenue (€)"]).values[0]
    value_da = abs(df.loc[df["Configuration"] == best_config, "DA arbitrage (€)"]).values[0]
    value_id = abs(df.loc[df["Configuration"] == best_config, "ID arbitrage (€)"]).values[0]
    value_as = abs(
        df.loc[df["Configuration"] == best_config, "FCR capacity revenue (€)"].values[0]
        + df.loc[df["Configuration"] == best_config, "aFRR capacity revenue (€)"].values[0]
        - df.loc[df["Configuration"] == best_config, "AS activation cost (€)"].values[0]
    )

    # ============================================================
    # 2) PIE CHART — WHERE DO YOUR SAVINGS COME FROM?
    # ============================================================

    st.subheader("🥧 Where Do Your Savings Come From? (Best Scenario)")

    pie_df = pd.DataFrame({
        "Component": ["PV", "DA Optimisation", "ID Optimisation", "Ancillary Services"],
        "Value": [value_pv, value_da, value_id, value_as]
    })

    pie_chart = (
        alt.Chart(pie_df)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="Value", type="quantitative"),
            color=alt.Color(field="Component", type="nominal"),
            tooltip=["Component", "Value"]
        )
        .properties(height=380)
    )

    st.altair_chart(pie_chart, use_container_width=True)

    st.markdown("""
    **Easy Reading:**  
    - The **bigger the slice**, the more that component reduces your annual electricity bill.  
    - PV normally provides the largest share.  
    - DA, ID, and AS add helpful extra value on top.
    """)

    st.markdown("---")

    # ============================================================
    # 3) BAR CHART — TOTAL ANNUAL COST PER SCENARIO (WITH LABELS)
    # ============================================================
    
    st.subheader("📉 Total Annual Cost per Scenario (Which One Is Best?)")
    
    # Compute savings vs PV-only
    pv_only_cost = df.loc[df["Configuration"] == "PV-only (No battery)", "Net annual cost (€)"].values[0]
    
    bar_data = df[["Configuration", "Net annual cost (€)"]].copy()
    bar_data["Savings vs PV-only (€)"] = pv_only_cost - bar_data["Net annual cost (€)"]
    
    # Altair chart with savings label
    base = alt.Chart(bar_data).encode(
        x=alt.X("Configuration:N", sort=None, title="Scenario"),
        y=alt.Y("Net annual cost (€):Q", title="Annual cost (€)"),
        tooltip=[
            "Configuration",
            "Net annual cost (€)",
            "Savings vs PV-only (€)"
        ]
    )
    
    bars = base.mark_bar().encode(
        color="Configuration:N"
    )
    
    labels = base.mark_text(
        dy=-10,  # position text above bar
        fontSize=14,
        fontWeight="bold",
        color="white"
    ).encode(
        text=alt.Text("Savings vs PV-only (€):Q", format="+.0f")
    )
    
    chart = (bars + labels).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("""
    **How to read this:**  
    - The **bigger the positive number**, the more money you save each year compared to PV-only.  
    - Example: If the label shows **+900 €**, that setup saves you 900 € more per year than PV-only.
    """)


    # ============================================================
    # 4) WATERFALL — HOW MUCH EACH OPTIMISATION STEP ADDS
    # ============================================================

    st.subheader("📈 How Much Each Step Adds (DA → ID → AS)")

    ordered = [
        "PV-only (No battery)",
        "Battery – non-optimised",
        "Battery – DA-optimised",
        "Battery – DA+ID-optimised",
        "Battery – DA+ID + AS (up-only)"
    ]
    ordered_existing = [c for c in ordered if c in df["Configuration"].tolist()]
    df_ord = df.set_index("Configuration").loc[ordered_existing].reset_index()

    df_ord["Value vs Grid (€)"] = c_grid - df_ord["Net annual cost (€)"]
    df_ord["Incremental gain (€)"] = df_ord["Value vs Grid (€)"].diff().fillna(
        df_ord["Value vs Grid (€)"].iloc[0]
    )

    waterfall_chart = (
        alt.Chart(df_ord)
        .mark_bar()
        .encode(
            x=alt.X("Configuration:N", sort=ordered_existing),
            y=alt.Y("Incremental gain (€):Q", title="Value Added (€)"),
            color="Configuration:N",
            tooltip=["Configuration", "Incremental gain (€)"]
        )
        .properties(height=350)
    )

    st.altair_chart(waterfall_chart, use_container_width=True)

    st.markdown("""
    **Easy Reading:**  
    - Each bar shows **how much extra** savings that step adds.  
    - DA adds value → ID adds a bit more → AS may add additional value.  
    - If any bar looks negative, your input assumptions are likely unrealistic.
    """)

    st.success(
        f"🎉 Best scenario: **{best_config}**, saving **{c_grid - c_best:,.0f} € / year**"
    )


# END OF BLOCK 6
# NEXT BLOCK = BLOCK 7 (Optimisation Logic + "How to understand results")
# ============================================================
# BLOCK 7 / N
# OPTIMISATION LOGIC (FULL) + HOW TO UNDERSTAND RESULTS
# ============================================================

with tabs[1]:

    st.header("🧠 Full Optimisation Logic (Restored Version)")

    st.markdown("""
    ## ☀️ 1. PV → Home → Battery (Physics-first dispatch)
    The model always follows this physical priority:

    **Step A — PV meets home load first**  
    - Every kWh of PV is used to cover your immediate consumption.  
    - This directly reduces grid import.  

    **Step B — Leftover PV goes into the battery**  
    - Only up to the battery’s theoretical throughput limit.  
    - Battery *never* exports to the grid.  

    **Step C — Any extra PV exports via EEG feed-in.**

    This ensures:
    - No illegal export-from-battery situations  
    - Everything is 100% EEG-compliant  
    - No unrealistic revenue inflations  
    """

    )

    st.markdown("""
    ---
    ## 🔋 2. Battery Throughput Model (Very simple & robust)
    We assume you can cycle your battery:

    ```
    cycles_per_day × batt_capacity × 365 × efficiency
    ```

    And then we apply a safety limiter:

    ```
    throughput × max_throughput_factor
    ```

    This means:
    - You can test optimistic cases (100%)  
    - Or conservative ones (60%)  
    """)

    st.markdown("""
    ---
    ## 🧮 3. DA Optimisation: The big value driver
    The model estimates an annualised *shiftable energy* equal to your usable
    battery throughput.

    Then:

    ```
    DA arbitrage = throughput × da_spread × capture_factor
    ```

    Where:
    - **da_spread** = typical *high-low* price range  
    - **capture_factor** = how much of this spread you realistically capture  
      (70% is a good average)

    **Interpretation:**  
    → The battery “charges” during cheap hours (modelled virtually)  
    → And “discharges” during expensive hours (modelled as avoided import)  
    """)

    st.markdown("""
    ---
    ## ⏱️ 4. ID Optimisation: The fine-tuning layer
    After DA is done, a smaller portion of that energy is passed through an
    intraday optimiser.

    ```
    ID arbitrage = (DA_energy × id_energy_factor) × id_spread × id_capture
    ```

    This makes sense because:
    - ID markets are more volatile  
    - Only a fraction of DA cycles can be “improved” intraday  
    - Capture factor is usually lower than DA  

    When ID value is high:
    - Your market assumptions are either very volatile  
    - Or very optimistic  
    """)

    st.markdown("""
    ---
    ## 🛰 5. Ancillary Services: FCR & aFRR (Up-only)
    To remain compliant + avoid grid export:

    - We model **upward** products only  
    - Battery **consumes** power when activated  
    - Battery **never** exports energy for FCR/aFRR  

    ### Revenue:
    ```
    Reserved_power(MW) × price(€/MW/h) × availability_hours
    ```

    ### Cost:
    ```
    Activation_energy × grid_price
    ```

    Activation energy is derived from:
    ```
    cycles × activation_factor × battery_throughput
    ```

    When AS is profitable:
    - Capacity price is strong  
    - Battery activation energy is low  
    - Inverter is right-sized  
    """)

    st.markdown("""
    ---
    ## 🎯 6. Annual Cost Calculation
    For every scenario:

    ```
    Net cost = Grid imports × price – EEG revenue
               – DA arbitrage – ID arbitrage
               – AS capacity revenue + AS activation cost
    ```

    Lower is better.  
    The “best” scenario is simply the row with the **lowest net cost**.
    """)


# ============================================================
# HOW TO UNDERSTAND RESULTS TAB
# ============================================================

with tabs[2]:

    st.header("📘 How to Understand These Results (Restored Deep Version)")

    st.markdown("""
    ## 🌞 1. If PV-only is already cheap → battery value comes mostly from optimisation
    If *PV-only (no battery)* is already much cheaper than grid-only:

    - Your PV system is sized well  
    - Any savings from the battery are “on top”  
    - DA optimisation becomes the dominant value source  

    **Battery value = DA + ID + flexibility**
    """)

    st.markdown("""
    ---
    ## 🔋 2. If “Battery – non-optimised” is not cheaper than PV-only
    This is normal.

    Base battery value *without optimisation* is usually weak or zero.

    Why?
    - You already self-consume PV  
    - Battery loses some energy due to roundtrip efficiency  
    - No price-based gains yet

    **Only when price optimisation is applied does the battery become profitable.**
    """)

    st.markdown("""
    ---
    ## ⚡ 3. DA Optimisation is usually the #1 driver
    If DA arbitrage is small, your spreads or capture factors are too low.

    If DA arbitrage is VERY high:
    - Check spreads  
    - Check capture factors  
    - Check cycles  
    """)

    st.markdown("""
    ---
    ## 🏎 4. If ID arbitrage is unexpectedly large
    ID should *never* dominate DA.

    If ID > DA:
    - You might have set **id_energy_factor** too high  
    - Or **id_spread** unrealistically high  
    - Or DA spreads too low  

    **Rule of thumb:**  
    - ID value ≈ 20–40% of DA value  
    """)

    st.markdown("""
    ---
    ## 🛰 5. How to interpret Ancillary Services (FCR/aFRR)
    - If your **AS value** is positive → great market, low activation  
    - If **AS value** is negative → activation is too high or prices too low  
    - If AS dwarfs DA+ID → parameters unrealistic  

    **Battery size & inverter limit matter massively.**
    """)

    st.markdown("""
    ---
    ## 🧪 6. If cost ordering breaks
    The normal ordering (from most to least expensive) is:

    ```
    Grid-only ≥ PV-only ≥ Battery non-opt ≥ DA-opt ≥ DA+ID ≥ DA+ID+AS
    ```

    If this breaks:
    - DA spreads wrong  
    - Battery parameters mismatched  
    - ID energy factor too aggressive  
    - AS unrealistic  

    The validation engine highlights exactly which part broke.
    """)

    st.markdown("""
    ---
    ## 🧠 7. General “sanity sense-checks”
    - PV generation should ≈ PV yield × kWp  
    - PV export should drop when battery size increases  
    - DA arbitrage should scale with throughput  
    - ID arbitrage should be smaller than DA  
    - AS revenue should not exceed battery value stack  
    - Net cost should never be **way** below zero  

    When everything behaves logically → trust the model.  
    """)

    st.success("🎉 You now have the full restored interpretation guide — identical depth, improved clarity.")

# END OF BLOCK 7
# NEXT BLOCK = BLOCK 8 (Model Assumptions + Limitations)
# ============================================================
# BLOCK 8 / N
# MODEL ASSUMPTIONS • MODEL LIMITATIONS (FULL RESTORED)
# ============================================================

# ------------------------------------------------------------
# MODEL ASSUMPTIONS TAB
# ------------------------------------------------------------
with tabs[3]:

    st.header("📚 Model Assumptions (Restored + Improved)")

    st.markdown("""
    This model is intentionally **simple**, **robust**, and **transparent**.  
    It avoids black-box behaviour and models only what truly matters.

    Below you’ll find every assumption used in FLEX.
    """)

    st.markdown("""
    ---
    ## ☀️ PV Model
    - PV generation = *kWp × annual yield (kWh/kWp)*
    - PV first covers home load (self-consumption-first logic)
    - Excess PV is either:
        - Stored in battery (if capacity available)
        - Or exported to the grid at EEG feed-in tariff
    - No complex degradation or shading model included
    - No hourly PV profile — annual energy balance model only (deliberate simplification)
    """)

    st.markdown("""
    ---
    ## 🔋 Battery Model
    - Battery **never exports** to grid (EEG-compliant model)
    - Battery usable energy yearly:

      ```
      cycles_per_day × battery_kWh × 365 × roundtrip_eff
      ```

    - Optional throughput limiter:

      ```
      × max_throughput_factor  (e.g., 100% = max possible)
      ```

    - Battery dispatch:
        - PV → Load → Battery (charge)
        - Battery → Load (discharge)
    - No SOC trajectory — annualised “energy bucket” method used
    - Roundtrip efficiency applied once on the throughput
    """)

    st.markdown("""
    ---
    ## 🧠 Price Optimisation (DA & ID)
    These use a **virtual shifting** approach — not hourly simulation.

    ### Day-Ahead (DA) Optimisation
    - Takes total battery throughput
    - Applies spread × capture factor
    - Represents arbitrage from cheap → expensive hours

    ### Intraday (ID) Optimisation
    - Only a *fraction* of DA cycles (ID energy factor)
    - Uses ID spread × ID capture factor
    - Represents fine-tuning beyond DA schedule
    """)

    st.markdown("""
    ---
    ## 🛰 Ancillary Services (FCR / aFRR)
    - Only **upward** reserve modelled (battery consumes power when activated)
    - No export → fully compliant with grid rules
    - Revenues:
        - Capacity payment × reserved MW × availability hours
    - Costs:
        - Extra charging energy (activation energy) × grid price
    - Activation energy estimated via:
      ```
      throughput × activation_factor × availability
      ```
    """)

    st.markdown("""
    ---
    ## 💶 Cost Model
    For each scenario:

    ```
    Net cost = grid_import_cost – EEG_revenue
               – DA_value – ID_value
               – AS_capacity_revenue + AS_activation_cost
    ```

    Everything is expressed as **net annual cost**.
    """)

    st.markdown("""
    ---
    ## 🔧 General Simplifications
    - Annual model (no hourly profile)
    - No battery degradation modelling
    - No dynamic tariffs or dynamic grid fees
    - No network constraints (DSO) included
    - Inverter efficiency not explicitly modelled (folded into battery eff)
    """)

    st.success("📘 All core assumptions restored — transparent and simple by design.")


# ------------------------------------------------------------
# MODEL LIMITATIONS TAB
# ------------------------------------------------------------
with tabs[4]:

    st.header("⚠️ Model Limitations (Restored + Improved)")

    st.markdown("""
    Despite being extremely useful for **scenario comparison**,  
    FLEX is **not** a full physical digital twin.

    Here are the known limitations:
    """)

    st.markdown("""
    ---
    ## 1. Annual-Only Model (no hourly resolution)
    The model uses **annual energy flows**, not hourly ones.

    🔸 What this means:  
    - No SOC curve  
    - No hour-by-hour scheduling  
    - No dynamic curtailment model  
    - No hourly PV clipping  

    ✔ This massively speeds up simulation  
    ✔ Ideal for strategic sizing & commercial value  
    ✘ Not suitable for grid-connection studies  
    """)

    st.markdown("""
    ---
    ## 2. No Battery Degradation
    - Useful for economic comparison  
    - But real batteries degrade over time  
    - High AS activation may increase degradation (not modelled)  
    """)

    st.markdown("""
    ---
    ## 3. No Dynamic Grid Tariffs
    - Grid price = one number  
    - No time-of-use Tarif-Zeitfenster  
    - No Strompreisbremse (price cap) logic  
    """)

    st.markdown("""
    ---
    ## 4. No Export-From-Battery
    This is intentional (EEG rules).  
    But in some countries you *can* export battery energy → FLEX disallows this.

    ➜ This means DA and ID optimisation always model **avoidance of imports**, not exports.
    """)

    st.markdown("""
    ---
    ## 5. No Grid Constraints or Power Limits
    - DSO restrictions  
    - Nodal constraints  
    - Charging peaks  
    - Inverter efficiency curves  

    None of these are modelled.
    """)

    st.markdown("""
    ---
    ## 6. AS Behaviour Simplified
    - All AS activation modelled as **energy taken from grid**  
    - Real AS has:
        - ramping  
        - activation delays  
        - symmetric vs asymmetric behaviour  
        - activation windows  
        - telemetry requirements  
    - FLEX abstracts these away into:
        - Activation factor  
        - Availability  
    """)

    st.markdown("""
    ---
    ## 7. Market Inputs Are User-Defined
    FLEX does not predict markets.  
    You choose spreads, capture factors, and activation.

    Wrong inputs = wrong result.
    """)

    st.warning("⚠️ These limitations are deliberate — FLEX is built for clarity, speed, and commercial insight, not grid physics.")


# END OF BLOCK 8
# NEXT: BLOCK 9 = AS explanation + Sensitivity tab
# ============================================================
# BLOCK 9 / N
# ANCILLARY SERVICES EXPLANATION + SENSITIVITY ANALYSIS
# ============================================================

# ------------------------------------------------------------
# TAB 7 — Ancillary Services Explained
# ------------------------------------------------------------
with tabs[5]:

    st.header("🛰 Ancillary Services Explained (Restored Full Version)")

    st.markdown("""
    Ancillary Services (AS) provide **stability** to the power grid.  
    Batteries can earn money by offering **flexibility** in the form of:

    - **FCR (Frequency Containment Reserve)**  
    - **aFRR (Automatic Frequency Restoration Reserve)**  

    FLEX models **only upward products** → battery **charges** when activated.  
    """)

    st.markdown("""
    ---
    ## ⚡ 1. Frequency Containment Reserve (FCR)
    FCR stabilises frequency deviations (50 Hz).

    - Activation is **fast** (milliseconds to seconds)  
    - Usually **short bursts**  
    - Payment is mostly **capacity-based**  
    - Activation energy is small but non-zero  

    In our model:
    ```
    FCR revenue = MW_reserved × price × availability_hours
    ```
    """)

    st.markdown("""
    ---
    ## 🔄 2. Automatic Frequency Restoration Reserve (aFRR)
    aFRR acts after FCR.

    - Activation slower (30 seconds → minutes)  
    - Longer duration  
    - Capacity prices lower than FCR  
    - Activation energy higher  

    Our model:
    ```
    aFRR revenue = MW_reserved × price × availability_hours
    ```
    """)

    st.markdown("""
    ---
    ## 🧮 3. Activation Energy (Upward only)
    Since we do not allow *export* from battery:

    - Every activation event = **battery consumes power**  
    - This draws energy from the **grid**  
    - Causing a cost  

    We model this simply:

    ```
    AS activation energy = throughput × activation_factor × availability
    ```
    """)

    st.markdown("""
    ---
    ## 🎯 When AS Makes Sense
    AS is attractive if:

    - Capacity prices are high  
    - Battery activation minimal  
    - Battery sized properly  
    - Inverter has enough headroom  

    It becomes less attractive if:
    - Activation too high  
    - AS revenue small  
    - Battery too small vs inverter  
    """)

    st.markdown("""
    ---
    ## ✔ Key Takeaways
    - AS can add a **third revenue stream**  
    - But only when modelling is realistic  
    - FLEX avoids unrealistic AS profits via validation checks  
    - Activation energy is always counted as cost  
    - Battery is always compliant → NO export  
    """)


# ------------------------------------------------------------
# TAB 8 — Sensitivity Analysis
# ------------------------------------------------------------
with tabs[6]:

    st.header("🧪 Sensitivity Analysis")

    st.markdown("""
    Use the sliders below to see **how sensitive** your cost outcome is to
    key parameters like DA spread, ID spread, capture factors, and activation.
    """)

    st.markdown("---")

    s_da = st.slider("DA spread sensitivity (€/kWh)", 0.05, 0.30, da_spread, 0.01)
    s_id = st.slider("ID spread sensitivity (€/kWh)", 0.05, 0.50, id_spread, 0.01)
    s_cap = st.slider("DA capture sensitivity", 0.2, 1.0, opt_cap, 0.05)
    s_idcap = st.slider("ID capture sensitivity", 0.2, 1.0, id_cap, 0.05)

    # --- FCR slider (safe version) ---
    fcr_min = 0.0
    fcr_max = 40.0
    
    # Safe default handling
    try:
        fcr_default = float(fcr_price_eur_per_mw_h)
        if not (fcr_min <= fcr_default <= fcr_max):
            fcr_default = 10.0
    except:
        fcr_default = 10.0
    
    s_fcr = st.slider(
        "FCR price sensitivity (€/MW/h)",
        min_value=fcr_min,
        max_value=fcr_max,
        value=fcr_default,
        step=1.0
    )

    # --- aFRR slider (safe version) ---
    afrr_min = 0.0
    afrr_max = 40.0
    
    # Safe default handling
    try:
        afrr_default = float(afrr_price_eur_per_mw_h)
        if not (afrr_min <= afrr_default <= afrr_max):
            afrr_default = 10.0
    except:
        afrr_default = 10.0
    
    s_afrr = st.slider(
        "aFRR price sensitivity (€/MW/h)",
        min_value=afrr_min,
        max_value=afrr_max,
        value=afrr_default,
        step=1.0
    )


    # Recompute scenario under sensitivity
    df_sens = compute_scenario(
        load_kwh, pv_kwp, pv_yield,
        grid_price, fit_price,
        batt_capacity, batt_eff, cycles, max_throughput_factor, sc_ratio,
        s_da, s_cap, nonopt_cap,
        s_id, s_idcap, id_energy_factor,
        as_enabled,
        fcr_power_kw, s_fcr, fcr_activation_factor,
        afrr_power_kw, s_afrr, afrr_activation_factor,
        as_availability_share
    )

    st.subheader("📊 Sensitivity Result")
    st.dataframe(df_sens, use_container_width=True)

    # Sensitivity chart — best-case net cost
    s_best = df_sens["Net annual cost (€)"].min()

    sens_chart = (
        alt.Chart(df_sens)
        .mark_bar()
        .encode(
            x="Configuration:N",
            y="Net annual cost (€):Q",
            color="Configuration:N",
            tooltip=["Configuration", "Net annual cost (€)"]
        )
        .properties(height=350)
    )

    st.altair_chart(sens_chart, use_container_width=True)

    st.success(
        f"📉 New best-case under sensitivity: {s_best:,.0f} € / yr"
    )

# END OF BLOCK 9
# NEXT BLOCK = BLOCK 10 (Live Market — ENTSO-E display)
# ============================================================
# BLOCK 10 / N
# LIVE MARKET TAB — ENTSO-E VISUALISATIONS
# ============================================================

with tabs[7]:

    st.header("📡 Live ENTSO-E Market Data")

    st.markdown("""
    This tab shows **real hourly Day-Ahead market prices** pulled directly from ENTSO-E
    (if enabled in the sidebar).
    
    Use it to:
    - Sense-check DA/ID spreads  
    - Understand market volatility  
    - Validate arbitrage assumptions  
    - Compare your inputs to reality  
    """)

    entsoe_series = st.session_state.get("entsoe_prices")

    if entsoe_series is None:
        st.info("ℹ️ Enable ENTSO-E live prices in the sidebar to show real market data.")
    else:
        st.subheader("📈 Hourly Day-Ahead Price (€/kWh)")

        df_live = (
            entsoe_series.rename("€/kWh")
            .reset_index()
            .rename(columns={"index": "datetime"})
        )

        chart = (
            alt.Chart(df_live)
            .mark_line(point=True)
            .encode(
                x=alt.X("datetime:T", title="Time"),
                y=alt.Y("€/kWh:Q", title="Price (€/kWh)"),
                tooltip=["datetime:T", "€/kWh:Q"]
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)

        # --------------------------------------------------------
        # PRICE METRICS
        # --------------------------------------------------------
        st.subheader("📊 Market Metrics")

        p_min = df_live["€/kWh"].min()
        p_max = df_live["€/kWh"].max()
        p_spread = p_max - p_min
        p_avg = df_live["€/kWh"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min price", f"{p_min:.3f} €")
        col2.metric("Max price", f"{p_max:.3f} €")
        col3.metric("Spread", f"{p_spread:.3f} €")
        col4.metric("Average", f"{p_avg:.3f} €")

        # --------------------------------------------------------
        # SHORTCUTS: 1-day, 3-day, 10-day windows
        # --------------------------------------------------------
        st.markdown("---")
        st.subheader("📅 Quick Time Windows")

        view = st.radio(
            "Select range",
            ["Last 24 hours", "Last 3 days", "Last 10 days"],
            horizontal=True
        )

        now = df_live["datetime"].max()

        if view == "Last 24 hours":
            df_view = df_live[df_live["datetime"] >= now - pd.Timedelta(days=1)]
        elif view == "Last 3 days":
            df_view = df_live[df_live["datetime"] >= now - pd.Timedelta(days=3)]
        else:
            df_view = df_live[df_live["datetime"] >= now - pd.Timedelta(days=10)]

        chart2 = (
            alt.Chart(df_view)
            .mark_line(point=True)
            .encode(
                x="datetime:T",
                y="€/kWh:Q",
                tooltip=["datetime:T", "€/kWh:Q"]
            )
            .properties(height=300)
        )

        st.altair_chart(chart2, use_container_width=True)

        # --------------------------------------------------------
        # INTERPRETATION NOTES
        # --------------------------------------------------------
        st.markdown("---")
        st.subheader("📘 How to interpret this")

        st.markdown("""
        - **Wide spreads** → Great for DA optimisation  
        - **Frequent spikes** → ID optimisation becomes more valuable  
        - **Long flat low periods** → Battery arbitrage weak  
        - **Low-volatility markets** → ID arbitrage usually minimal  
        - **High max-min spread** → Big arbitrage potential  
        """)

        st.success("📡 Live ENTSO-E data loaded successfully.")

# END OF BLOCK 10
# NEXT BLOCK = BLOCK 11 (Multi-audience summary)
# ============================================================
# BLOCK 11 / N
# MULTI-AUDIENCE SUMMARY (FULL RESTORED VERSION)
# ============================================================

with tabs[8]:

    st.header("👥 Multi-Audience Summary")

    c_grid = load_kwh * grid_price
    c_best = df["Net annual cost (€)"].min()
    best_config = df.loc[df["Net annual cost (€)"].idxmin(), "Configuration"]
    savings = c_grid - c_best

    # Pull some key metrics for explanation
    pv_gen = df.loc[0, "PV generation (kWh)"]
    batt_da = df["DA arbitrage (€)"].max()
    batt_id = df["ID arbitrage (€)"].max()
    as_rev = (
        df.get("FCR capacity revenue (€)", pd.Series([0])).max()
        + df.get("aFRR capacity revenue (€)", pd.Series([0])).max()
    )
    as_cost = df.get("AS activation cost (€)", pd.Series([0])).max()
    as_net = as_rev - as_cost

    st.markdown("### 🎯 Homeowner Summary (Simple, Non-technical)")
    st.markdown(f"""
    **Your cheapest option is:**  
    👉 **{best_config}**

    **Annual savings:**  
    💰 **{savings:,.0f} € per year**

    **What this means:**  
    - Your PV generates ~**{pv_gen:,.0f} kWh/year**  
    - The battery reduces imports and buys electricity when cheap  
    - You save money mainly through **smart timing**, not just storage  
    - Everything happens automatically (no daily adjustments)  
    """)

    st.markdown("---")

    st.markdown("### 🛠 Installer / EPC Summary")
    st.markdown(f"""
    **Most economically efficient system:**  
    📦 **{best_config}**

    **Key drivers:**  
    - PV yield determines baseline economics  
    - Battery throughput × market spreads determine optimisation value  
    - Battery → Self-consumption → Arbitrage → AS value stack  
    - No over-dimensioning penalties (model highlights if unrealistic)

    **Recommended system guidance:**  
    - Ensure inverter limit ≥ expected FCR/aFRR bids  
    - Prefer batteries with ≥1 full-cycle/day durability  
    - Oversizing PV often helps even without optimisation  
    """)

    st.markdown("---")

    st.markdown("### 📈 Trader / Aggregator Summary (Market-facing)")
    st.markdown(f"""
    **DA arbitrage value:** {batt_da:,.0f} € / yr  
    **ID arbitrage value:** {batt_id:,.0f} € / yr  
    **AS contribution:** {as_net:,.0f} € / yr (net)

    **Interpretation:**  
    - DA remains the **primary PnL driver**  
    - ID adds a **non-trivial but smaller layer**  
    - AS is **context-dependent**: profitable only with low activation  
    - Battery never exports — all optimisation is compliant  

    **Operational suggestion:**  
    - Low-volatility → reduce ID / AS bids  
    - High-volatility → increase ID exposure  
    - High AS prices → reserve more capacity  
    """)

    st.markdown("---")

    st.markdown("### 🏛 Regulator / Policy Summary")
    st.markdown(f"""
    **This configuration cuts grid demand by:**  
    🔌 ~{(pv_gen/load_kwh)*100:.1f}% of PV generation used behind the meter.

    **Grid benefits:**  
    - Reduced peak imports  
    - More flexible load shifting  
    - More FCR/aFRR capacity available  
    - No battery export → avoids tariff distortions  

    **Policy implication:**  
    - Smart batteries amplify PV grid relief  
    - Availability-based AS models avoid overcompensation  
    - DA/ID optimisation reduces stress on peak hours  
    """)

    st.markdown("---")

    st.markdown("### 🧪 Engineer / Technical Summary")
    st.markdown("""
    **Model characteristics:**  
    - Energy-based annual model (not time-series)  
    - PV-first dispatch, no-export battery  
    - DA/ID arbitrage proportional to throughput and spreads  
    - AS activation energy proportional to throughput × activation  
    - All flows validated through physics and economics checks  

    **Useful for:**  
    - Sizing studies  
    - Business case design  
    - Market participation strategies  

    **Not suitable for:**  
    - Protection studies  
    - Transformer hosting capacity analysis  
    - Dynamic network behaviour  
    """)

    st.success("👥 Multi-audience summary fully restored.")
# END OF BLOCK 11
# NEXT BLOCK = BLOCK 12 (Admin Panel)
# ============================================================
# BLOCK 12 / N  — FINAL BLOCK
# ADMIN PANEL (Debug, Reset, Export, Dev Tools)
# ============================================================
# ============================================================
# REQUIRED VARIABLES FOR COMPETITOR BENCHMARK SECTION
# (Fixes NameError for PV size, battery size, best_config, etc.)
# ============================================================

# 1 — Retrieve best optimisation configuration
best_config = df.loc[df["Net annual cost (€)"].idxmin()]

# 2 — Extract PV and Battery sizes (fallbacks added)
pv_kwp = best_config["PV capacity (kWp)"] if "PV capacity (kWp)" in best_config else 0
battery_capacity_kwh = best_config["Battery capacity (kWh)"] if "Battery capacity (kWh)" in best_config else 0

# 3 — Compute your optimised annual savings
my_best_cost = best_config["Net annual cost (€)"]
my_best_savings = load_kwh * grid_price - my_best_cost

import datetime as dt

# ------------------------------------------------------------
# COMPETITOR BENCHMARK TAB (CLEAN & FIXED)
# ------------------------------------------------------------
with tabs[9]:

    st.header("🏆 Competitor Benchmarking")

    # --------------------------------------------------------
    # 1. Load competitors.json (auto-detected offers)
    # --------------------------------------------------------
    import json, os, math

    json_path = "competitors.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                auto_competitors = json.load(f)
                if not isinstance(auto_competitors, list):
                    auto_competitors = []
        except Exception:
            auto_competitors = []
            st.warning("competitors.json exists but could not be read.")
    else:
        auto_competitors = []
        st.info("No competitors.json file found yet – it will be created by the alert system when new offers are saved.")

    # --------------------------------------------------------
    # 2. Show automatically detected competitors
    # --------------------------------------------------------
    st.subheader("🔄 Automatically Detected Competitor Offers")

    if len(auto_competitors) == 0:
        st.info("No automatically detected competitors yet.")
        auto_df = pd.DataFrame([])
    else:
        auto_df = pd.DataFrame(auto_competitors)

        # Make sure column names match the benchmark expectations
        auto_df = auto_df.rename(columns={
            "name": "Competitor Name",
            "pv_size": "PV Size (kWp)",
            "battery_size": "Battery Size (kWh)",
            "claimed_savings": "Claimed Annual Savings (€)"
        })

        for col in ["Competitor Name", "PV Size (kWp)", "Battery Size (kWh)", "Claimed Annual Savings (€)"]:
            if col not in auto_df.columns:
                auto_df[col] = 0.0

        st.dataframe(auto_df, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # 3. User-entered competitors
    # --------------------------------------------------------
    st.subheader("🧩 Add Your Own Competitor Offers")

    manual_template = pd.DataFrame({
        "Competitor Name": [],
        "PV Size (kWp)": [],
        "Battery Size (kWh)": [],
        "Claimed Annual Savings (€)": []
    })

    competitor_df = st.data_editor(
        manual_template,
        num_rows="dynamic",
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 4. Merge FLEX + auto + manual competitors
    # --------------------------------------------------------

    # FLEX model entry (from optimisation)
    flex_entry = {
        "Competitor Name": "Your FLEX Optimised Model",
        "PV Size (kWp)": float(flex_pv_kwp),
        "Battery Size (kWh)": float(flex_batt_kwh),
        "Claimed Annual Savings (€)": float(my_best_savings),
    }

    flex_df = pd.DataFrame([flex_entry])

    # Merge auto + manual
    if len(auto_df) > 0:
        combined_df = pd.concat([competitor_df, auto_df], ignore_index=True, sort=False)
    else:
        combined_df = competitor_df.copy()

    combined_df = combined_df.fillna(0)

    # Full dataset including FLEX
    if len(combined_df) > 0:
        score_df = pd.concat([flex_df, combined_df], ignore_index=True, sort=False)
    else:
        score_df = flex_df.copy()

    # Display combined table (without internal norm columns)
    st.subheader("📦 Combined Benchmark Dataset")
    st.dataframe(
        score_df[["Competitor Name", "PV Size (kWp)", "Battery Size (kWh)", "Claimed Annual Savings (€)"]],
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 5. Score Index (normalised comparison)
    # --------------------------------------------------------
    st.subheader("🏅 Score Index & Ranking")

    if len(score_df) == 0:
        st.info("No competitors to score yet.")
    else:
        score_df = score_df.fillna(0)

        def _norm(col):
            mn = score_df[col].min()
            mx = score_df[col].max()
            if mx - mn == 0:
                return pd.Series(1.0, index=score_df.index)
            return (score_df[col] - mn) / (mx - mn)

        score_df["PV_norm"] = _norm("PV Size (kWp)")
        score_df["Batt_norm"] = _norm("Battery Size (kWh)")
        score_df["Savings_norm"] = _norm("Claimed Annual Savings (€)")

        score_df["Score Index"] = (
            0.3 * score_df["PV_norm"] +
            0.3 * score_df["Batt_norm"] +
            0.4 * score_df["Savings_norm"]
        )

        score_sorted = score_df.sort_values("Score Index", ascending=False).reset_index(drop=True)

        st.dataframe(
            score_sorted[[
                "Competitor Name",
                "PV Size (kWp)",
                "Battery Size (kWh)",
                "Claimed Annual Savings (€)",
                "PV_norm",
                "Batt_norm",
                "Savings_norm",
                "Score Index"
            ]],
            use_container_width=True
        )

        # Bar chart ranking
        rank_chart = (
            alt.Chart(score_sorted)
            .mark_bar()
            .encode(
                x=alt.X("Competitor Name:N", sort=score_sorted["Competitor Name"].tolist()),
                y=alt.Y("Score Index:Q"),
                color="Competitor Name:N",
                tooltip=["Competitor Name", "Score Index"]
            )
            .properties(height=350)
        )
        st.altair_chart(rank_chart, use_container_width=True)

        best_comp = score_sorted.iloc[0]
        worst_comp = score_sorted.iloc[-1]

        st.success(f"🥇 Best: {best_comp['Competitor Name']} — Score {best_comp['Score Index']:.2f}")
        st.warning(f"🥉 Lowest: {worst_comp['Competitor Name']} — Score {worst_comp['Score Index']:.2f}")

        st.markdown("---")

        # ----------------------------------------------------
        # 6. FLEX vs competitors gap chart (€/year)
        # ----------------------------------------------------
        st.subheader("📉 FLEX vs Competitors – €/year Gap Analysis")

        gap_df = combined_df.copy()
        if len(gap_df) > 0:
            gap_df["Your FLEX Savings (€)"] = float(my_best_savings)
            gap_df["Gap (€)"] = gap_df["Your FLEX Savings (€)"] - gap_df["Claimed Annual Savings (€)"]

            gap_chart = (
                alt.Chart(gap_df)
                .mark_bar()
                .encode(
                    x=alt.X("Competitor Name:N", sort="-y"),
                    y=alt.Y("Gap (€):Q"),
                    color=alt.condition(
                        alt.datum["Gap (€)"] > 0,
                        alt.value("#10b981"),  # green FLEX better
                        alt.value("#ef4444"),  # red competitor better
                    ),
                    tooltip=[
                        "Competitor Name",
                        "Claimed Annual Savings (€)",
                        "Your FLEX Savings (€)",
                        "Gap (€)"
                    ]
                )
                .properties(height=300)
            )
            st.altair_chart(gap_chart, use_container_width=True)

            st.markdown("""
            **How to read this chart:**
            - Green bars → FLEX has higher annual savings than the competitor.  
            - Red bars → Competitor's claimed savings are higher than FLEX.  
            - The bar height is the €/year difference.
            """)
        else:
            st.info("Add at least one competitor to see a gap chart.")

        st.markdown("---")

        # ----------------------------------------------------
        # 7. Simple radar chart (normalised PV / Battery / Savings)
        # ----------------------------------------------------
        # ------------------------------------------------------------
    # SIMPLE RADAR CHART (Plotly version)
    # ------------------------------------------------------------
    
    import plotly.graph_objects as go
    
    if len(combined_df) == 0:
        st.info("Add manual competitors or wait for automatic competitor detection to generate a radar chart.")
    else:
    
        # Build radar dataset
        radar_df = combined_df.copy()
    
        # Add FLEX model as first entry
        my_entry = {
            "Competitor Name": "Your FLEX Optimised Model",
            "PV Size (kWp)": float(best_config["PV capacity (kWp)"]) if "PV capacity (kWp)" in best_config else float(pv_kwp),
    "Battery Size (kWh)": float(best_config["Battery capacity (kWh)"]) if "Battery capacity (kWh)" in best_config else 0,
            "Claimed Annual Savings (€)": float(my_best_savings),
        }
    
        radar_df = pd.concat([pd.DataFrame([my_entry]), radar_df], ignore_index=True)
        radar_df = radar_df.fillna(0)
    
        # Normalise values 0–1
        def norm(series):
            mn, mx = series.min(), series.max()
            if mx - mn == 0:
                return [1] * len(series)
            return (series - mn) / (mx - mn)
    
        radar_df["PV_norm"] = norm(radar_df["PV Size (kWp)"])
        radar_df["Batt_norm"] = norm(radar_df["Battery Size (kWh)"])
        radar_df["Savings_norm"] = norm(radar_df["Claimed Annual Savings (€)"])
    
        categories = ["PV_norm", "Batt_norm", "Savings_norm"]
    
        # Plotly radar chart
        fig = go.Figure()
    
        for i, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories],
                theta=["PV Size", "Battery Size", "Annual Savings"],
                fill='toself',
                name=row["Competitor Name"]
            ))
    
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            height=600
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("""
        **How to read this chart:**  
        - Bigger filled area = stronger competitor.  
        - All values are normalised 0–1 for fair comparison.  
        - FLEX model is shown alongside real competitor propositions.
        """)




# ------------------------------------------------------------
# STEP 4 — COMPETITOR ALERTS INBOX TAB
# ------------------------------------------------------------
with tabs[10]:
    st.header("📨 Competitor Alerts Inbox")

    st.markdown("""
    This inbox shows all **new competitor offers** discovered automatically by your monitoring system.
    Whenever a new E.ON, Octopus, Sonnen, EnBW, Senec or Tibber PV+Battery offer appears,
    it will show up here with all details.
    """)

    # The auto_competitors list is already loaded in Step 2 (competitors.json)
    if len(auto_competitors) == 0:
        st.info("No competitor alerts yet — new offers will appear here automatically as soon as they are detected.")
    else:
        st.success(f"{len(auto_competitors)} competitor offers detected!")

        for comp in auto_competitors:
            with st.expander(f"📌 {comp.get('name', 'Unknown Offer')} (detected {comp.get('detected_on', 'N/A')})"):

                st.write(f"**Competitor:** {comp.get('name', 'N/A')}")
                st.write(f"**PV Size:** {comp.get('pv_size', 'N/A')} kWp")
                st.write(f"**Battery Size:** {comp.get('battery_size', 'N/A')} kWh")
                st.write(f"**Claimed Savings:** {comp.get('claimed_savings', 'N/A')} € / year")

                source = comp.get("source", None)
                if source:
                    st.markdown(f"**Source:** [{source}]({source})")

                st.markdown("---")
                st.caption("This offer has been automatically added to your Competitor Benchmark dataset.")







with tabs[11]:

    st.header("🛠 Admin Panel (Advanced Controls)")

    st.markdown("""
    The Admin Panel provides:
    - Session reset  
    - Debugging tools  
    - Parameter dumps  
    - Export / download of scenario results  
    - Developer-level toggles  
    """)

    # --------------------------------------------------------
    # 1. SESSION RESET
    # --------------------------------------------------------
    st.subheader("🔄 Reset Session")

    if st.button("🧹 Reset all inputs & session state"):
        st.session_state.clear()
        st.success("Session cleared. Please reload the page.")
        st.stop()

    st.markdown("---")

    # --------------------------------------------------------
    # 2. EXPORT RESULTS (JSON + CSV)
    # --------------------------------------------------------
    st.subheader("📦 Export Results")

    # JSON export
    export_json = df.to_json(orient="records", indent=2)
    st.download_button(
        label="📥 Download results (JSON)",
        data=export_json,
        file_name="flex_results.json",
        mime="application/json"
    )

    # CSV export
    export_csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download results (CSV)",
        data=export_csv,
        file_name="flex_results.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 3. PARAMETER DUMP (Everything in one place)
    # --------------------------------------------------------
    st.subheader("📝 Parameter Dump")

    params_dump = {
        "load_kwh": load_kwh,
        "pv_kwp": pv_kwp,
        "pv_yield": pv_yield,
        "grid_price": grid_price,
        "fit_price": fit_price,
        "battery_capacity": batt_capacity,
        "battery_efficiency": batt_eff,
        "cycles_per_day": cycles,
        "max_throughput_factor": max_throughput_factor,
        "self_consumption_ratio_no_batt": sc_ratio,
        "da_spread": da_spread,
        "da_capture_opt": opt_cap,
        "da_capture_nonopt": nonopt_cap,
        "id_spread": id_spread,
        "id_capture": id_cap,
        "id_energy_factor": id_energy_factor,
        "as_enabled": as_enabled,
        "fcr_power_kw": fcr_power_kw,
        "fcr_price_eur_per_mw_h": fcr_price_eur_per_mw_h,
        "fcr_activation_factor": fcr_activation_factor,
        "afrr_power_kw": afrr_power_kw,
        "afrr_price_eur_per_mw_h": afrr_price_eur_per_mw_h,
        "afrr_activation_factor": afrr_activation_factor,
        "as_availability_share": as_availability_share
    }

    st.json(params_dump)

    st.markdown("---")

    # --------------------------------------------------------
    # 4. INTERNAL DEBUG INSPECTOR
    # --------------------------------------------------------
    st.subheader("🧑‍💻 Internal Debug Inspector")

    if st.checkbox("Show full session_state"):
        st.write(st.session_state)

    if st.checkbox("Show raw scenario DataFrame"):
        st.dataframe(df, use_container_width=True)

    if st.checkbox("Show DataFrame dtypes"):
        st.write(df.dtypes)

    st.markdown("---")

    # --------------------------------------------------------
    # 5. ADVANCED DEVELOPER TOGGLES
    # --------------------------------------------------------
    st.subheader("⚙️ Developer Settings")

    dev_verbose = st.checkbox("Enable verbose engine logging", value=False)
    dev_trace = st.checkbox("Trace AS activation calculation", value=False)
    dev_internal_checks = st.checkbox("Show internal validators", value=False)

    if dev_verbose:
        st.info("Verbose mode ON — engine will print intermediate steps to the logs.")

    if dev_trace and as_enabled:
        st.info(f"""
        ### AS Activation Trace
        - FCR power: {fcr_power_kw} kW  
        - aFRR power: {afrr_power_kw} kW  
        - FCR activation factor: {fcr_activation_factor}  
        - aFRR activation factor: {afrr_activation_factor}  
        - Availability: {as_availability_share*100:.1f}%  
        """)

    if dev_internal_checks:
        st.info("Internal check: DataFrame columns")
        st.write(df.columns)

    st.success("🛠 Admin tools loaded successfully — all blocks assembled.")

# END OF BLOCK 12 — FULL APPLICATION COMPLETE
