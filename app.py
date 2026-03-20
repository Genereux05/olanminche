"""
Olanminché — Dashboard Interactif
Détection et prédiction de la Maladie Rénale Chronique
Bootcamp Cohorte 1 - 
"""

import os
import sys
import json
import hmac
import hashlib
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.preprocessing import (
    load_and_clean, get_feature_matrix,
    eGFR_CKD_EPI, egfr_to_ckd_stage,
    compute_risk_score, FEATURE_COLS, FEATURE_LABELS
)
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olanminché — Tableau de Bord",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
    --bg-main: #f3f6fb;
    --bg-soft: #eaf0f7;
    --surface: #ffffff;
    --ink: #000000;
    --muted: #000000;
    --line: #d4deea;
    --primary: #0b57d0;
    --primary-deep: #0842a0;
    --accent: #0f766e;
    --warn: #b45309;
    --danger: #b91c1c;
}

@keyframes fadeLift {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    color: var(--ink);
    font-weight: 500;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
    color: #001a72 !important;
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] small,
[data-testid="stAppViewContainer"] .stCaption,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 520px at 2% -8%, #dce9ff 0%, transparent 58%),
        radial-gradient(780px 420px at 100% 0%, #d7f2eb 0%, transparent 55%),
        linear-gradient(180deg, var(--bg-main) 0%, var(--bg-soft) 100%);
}

[data-testid="stHeader"] { background: transparent; }

.main-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(118deg, #0f172a 0%, #0b57d0 58%, #1976d2 100%);
    padding: 2rem 2.2rem;
    border-radius: 22px;
    margin-bottom: 1.3rem;
    color: #f8fafc;
    box-shadow: 0 20px 45px rgba(11, 87, 208, 0.24);
    animation: fadeLift 420ms ease-out both;
}

.main-header::after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -80px;
    top: -120px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.14);
}

.main-header h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.4rem 0 0;
    opacity: 0.96;
    font-size: 1rem;
}

.page-hero {
    background: linear-gradient(140deg, #ffffff 0%, #edf3ff 50%, #eefaf4 100%);
    border: 1px solid #d8e2f0;
    border-radius: 20px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    animation: fadeLift 380ms ease-out both;
}

.page-hero h2 {
    margin: 0;
    color: #0f172a;
    font-size: 1.58rem;
    font-weight: 700;
}

.page-hero p {
    margin: 0.35rem 0 0;
    color: #000000;
    font-weight: 500;
}

.section-title {
    font-size: 1.06rem;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #cfdced;
    padding-bottom: 0.4rem;
    margin: 0.9rem 0;
}

.metric-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border: 1px solid #d8e3ef;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
    border-top: 4px solid var(--primary);
    margin-bottom: 1rem;
    animation: fadeLift 460ms ease-out both;
}

.metric-card.green  { border-top-color: var(--accent); }
.metric-card.orange { border-top-color: var(--warn); }
.metric-card.red    { border-top-color: var(--danger); }
.metric-card.blue   { border-top-color: var(--primary); }

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.95rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}

.metric-label {
    font-size: 0.79rem;
    color: #000000;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fbfdff 0%, #eef3f9 100%);
    border-right: 1px solid #dae3ee;
}

[data-testid="stSidebar"] * {
    color: #000000 !important;
    font-weight: 500;
}

div[data-testid="stForm"],
div[data-testid="stExpander"],
div[data-testid="stDataFrame"],
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d9e2ed;
    border-radius: 14px;
}

div[data-testid="stDataFrame"] { padding: 0.2rem; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    border-radius: 12px !important;
    border-color: #c8d6e6 !important;
    background-color: #ffffff !important;
}

div[data-baseweb="input"] input,
input[type="text"],
input[type="password"],
textarea {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    caret-color: #0f172a !important;
}

div[data-baseweb="input"] input::placeholder,
input[type="text"]::placeholder,
input[type="password"]::placeholder,
textarea::placeholder {
    color: #000000 !important;
    opacity: 0.6 !important;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--primary);
    background: linear-gradient(110deg, #0b57d0 0%, #1a73e8 100%);
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.stButton > button:hover {
    border-color: var(--primary-deep);
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(11, 87, 208, 0.25);
}

button[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    border: 1px solid #d5dfeb !important;
    background: #f7faff !important;
    color: #000000 !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, #eef4ff, #edf9f3) !important;
    color: #000000 !important;
    border-bottom-color: #0b57d0 !important;
    font-weight: 700 !important;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #d9e2ed;
    border-radius: 14px;
    padding: 0.25rem;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
}

.login-shell {
    max-width: 980px;
    margin: 2.1rem auto 0.8rem;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #d6e2f0;
    border-radius: 22px;
    box-shadow: 0 20px 42px rgba(15, 23, 42, 0.12);
    overflow: hidden;
}

.login-left {
    background: linear-gradient(125deg, #0f172a 0%, #0b57d0 75%, #1976d2 100%);
    color: #f8fafc;
    padding: 2rem 1.8rem;
    border-radius: 16px;
    min-height: 300px;
}

.login-left h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
}

.login-left p {
    margin-top: 0.6rem;
    color: rgba(248, 250, 252, 0.95);
}

.login-badge {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.34rem 0.7rem;
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.login-right {
    padding: 1.4rem 1.4rem 1.2rem;
}

.login-right h3 {
    margin: 0;
    font-size: 1.15rem;
    color: #0f172a;
}

.login-note {
    margin: 0.35rem 0 1rem;
    color: #000000;
    font-size: 0.94rem;
    font-weight: 500;
}

.sidebar-card {
    border: 1px solid #d5dfeb;
    border-radius: 14px;
    padding: 0.75rem 0.8rem;
    background: #ffffff;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    margin-top: 0.85rem;
    animation: fadeLift 500ms ease-out both;
}

.sidebar-brand {
    padding: 0.9rem 0.6rem 0.4rem;
    text-align: center;
}

.sidebar-brand h2 {
    margin: 0.3rem 0 0;
    font-size: 1.22rem;
}

.sidebar-brand p {
    margin: 0.2rem 0 0;
    color: #000000;
    font-size: 0.78rem;
    font-weight: 600;
}

@media (max-width: 900px) {
    .login-shell {
        margin: 1rem 0 0.5rem;
        border-radius: 16px;
    }

    .login-left,
    .login-right {
        padding: 1.2rem 1rem;
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTES GÉO BÉNIN
# ─────────────────────────────────────────────────────────
BENIN_GEO = {
    'Littoral':   (6.3703,  2.3912),
    'Atlantique': (6.4000,  2.3500),
    'Ouémé':      (6.5500,  2.6500),
    'Plateau':    (7.2000,  2.6000),
    'Mono':       (6.8000,  1.6000),
    'Couffo':     (7.1000,  1.7500),
    'Zou':        (7.2000,  2.2000),
    'Collines':   (8.2000,  2.2000),
    'Borgou':     (9.3000,  2.7000),
    'Alibori':    (11.2000, 2.7000),
    'Atacora':    (10.5000, 1.6500),
    'Donga':      (9.7000,  1.7000),
}

STAGE_COLORS = {
    1: '#2e7d32', 2: '#1565c0', 3: '#f9a825', 4: '#e65100', 5: '#c62828'
}
STAGE_NAMES = {
    1: 'CKD 1 (Normal)', 2: 'CKD 2 (Légère)', 3: 'CKD 3 (Modérée)',
    4: 'CKD 4 (Sévère)', 5: 'CKD 5 (Terminale)'
}
RISK_COLORS = {'Faible': '#2e7d32', 'Modéré': '#f57f17', 'Élevé': '#c62828'}
PLOT_TEXT_COLOR = "#001a72"

# Thème global Plotly pour rendre tous les textes lisibles.
PLOTLY_THEME = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Manrope", color=PLOT_TEXT_COLOR),
        title=dict(font=dict(color=PLOT_TEXT_COLOR)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title_font=dict(color=PLOT_TEXT_COLOR),
            tickfont=dict(color=PLOT_TEXT_COLOR),
            color=PLOT_TEXT_COLOR,
            gridcolor="#dce8f4",
            zerolinecolor="#c8d8ea",
        ),
        yaxis=dict(
            title_font=dict(color=PLOT_TEXT_COLOR),
            tickfont=dict(color=PLOT_TEXT_COLOR),
            color=PLOT_TEXT_COLOR,
            gridcolor="#dce8f4",
            zerolinecolor="#c8d8ea",
        ),
        legend=dict(font=dict(color=PLOT_TEXT_COLOR), title=dict(font=dict(color=PLOT_TEXT_COLOR))),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color=PLOT_TEXT_COLOR),
                title=dict(font=dict(color=PLOT_TEXT_COLOR))
            )
        ),
        polar=dict(
            angularaxis=dict(tickfont=dict(color=PLOT_TEXT_COLOR)),
            radialaxis=dict(tickfont=dict(color=PLOT_TEXT_COLOR), gridcolor="#dce8f4"),
        ),
    )
)
px.defaults.template = PLOTLY_THEME

# ─────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES & MODÈLES
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = os.path.join(ROOT, 'data', 'Data_AI4CKD_Original.csv')
    return load_and_clean(path)

@st.cache_resource(show_spinner=False)
def load_models():
    model_dir = os.path.join(ROOT, 'models')
    models = {}
    for name in ['rf_model', 'gb_model', 'best_model']:
        p = os.path.join(model_dir, f'{name}.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                mdl = pickle.load(f)
                models[name] = patch_simple_imputer_compat(mdl)
    meta_path = os.path.join(model_dir, 'metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return models, meta


def page_hero(title, subtitle):
    st.markdown(
        f"""
        <div class='page-hero'>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def patch_simple_imputer_compat(estimator):
    """Patch legacy SimpleImputer objects loaded from older sklearn pickles."""
    visited = set()

    def _walk(obj):
        oid = id(obj)
        if obj is None or oid in visited:
            return
        visited.add(oid)

        if isinstance(obj, SimpleImputer):
            if not hasattr(obj, "_fill_dtype"):
                stats = getattr(obj, "statistics_", None)
                if stats is not None:
                    obj._fill_dtype = np.asarray(stats).dtype
                else:
                    obj._fill_dtype = np.dtype("float64")
            return

        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return

        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                _walk(v)
            return

        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                _walk(v)

    _walk(estimator)
    return estimator


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def authenticate_user(username, password):
    env_user = os.getenv("APP_USERNAME", "admin")
    env_hash = os.getenv("APP_PASSWORD_HASH", "").strip()
    env_plain = os.getenv("APP_PASSWORD", "").strip()

    if username != env_user:
        return False
    if env_hash:
        return hmac.compare_digest(_sha256(password), env_hash)
    if env_plain:
        return hmac.compare_digest(password, env_plain)
    # Fallback dev-only password if no env vars are configured.
    return hmac.compare_digest(password, "ChangeMeNow123!")

# ─────────────────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_user" not in st.session_state:
    st.session_state.auth_user = ""

if not st.session_state.authenticated:
    st.markdown("<div class='login-shell'>", unsafe_allow_html=True)
    login_left, login_right = st.columns([1.25, 1], gap="large")

    with login_left:
        st.markdown("""
        <div class='login-left'>
            <span class='login-badge'>Plateforme Clinique</span>
            <h1>Olanminche Intelligence Care</h1>
            <p>Un espace unifie pour l'analyse CKD, la prediction et l'aide a la decision clinique.</p>
            <p style='margin-top:1.2rem; font-size:0.9rem; opacity:.9;'>Concu pour les equipes medicales, data et pilotage hospitalier.</p>
        </div>
        """, unsafe_allow_html=True)

    with login_right:
        st.markdown("""
        <div class='login-right'>
            <h3>Connexion securisee</h3>
            <p class='login-note'>Utilisez vos identifiants administrateur pour acceder au dashboard complet.</p>
        </div>
        """, unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            login_user = st.text_input("Identifiant", placeholder="ex: admin.centre")
            login_pass = st.text_input("Mot de passe", type="password", placeholder="Votre mot de passe")
            login_submit = st.form_submit_button("Acceder a la plateforme", use_container_width=True, type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if login_submit:
        if authenticate_user(login_user, login_pass):
            st.session_state.authenticated = True
            st.session_state.auth_user = login_user
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
        else:
            st.error("Identifiant ou mot de passe invalide.")

    st.stop()

# ─────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class='sidebar-brand'>
            <span style='font-size:2rem;'>📊</span>
            <h2>Olanminche</h2>
            <p>Clinical Intelligence Platform</p>
    </div>
        <hr style='border-color:#d4dfeb; margin: 0.4rem 0 0.8rem;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Accueil & Vue d'ensemble",
         "📊 Analyse Exploratoire",
         "🗺️ Cartographie des Risques",
         "🤖 Prédiction IA",
         "📈 Performance des Modèles",
         "👁️ Explications & SHAP"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#d4dfeb;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-card' style='font-size:0.78rem;'>
    <b>Session:</b> utilisateur connecte<br>
    <b>Modeles ML:</b> Random Forest, Gradient Boosting<br>
    <b>Dataset:</b> 309 patients<br>
    <b>Variables:</b> 30 features<br>
    <b>F1 Score:</b> 0.897
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Connecté: {st.session_state.get('auth_user', 'N/A')}")
    if st.button("Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_user = ""
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# ─────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────
with st.spinner("Chargement des données..."):
    df = load_data()
    models, meta = load_models()

# ─────────────────────────────────────────────────────────
# ① ACCUEIL & VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────
if page == "🏠 Accueil & Vue d'ensemble":
    st.markdown("""
    <div class='main-header'>
      <h1>Olanminché</h1>
      <p>Plateforme visuelle premium de détection et de prédiction de la Maladie Rénale Chronique</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────
    total = len(df)
    stage_counts = df['stage'].value_counts().sort_index()
    high_risk = (df['risk_category'] == 'Élevé').sum()
    ckd45 = (df['stage'] >= 4).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, total, "Patients Total", "blue"),
        (c2, int(stage_counts.get(1, 0) + stage_counts.get(2, 0)), "CKD Stades 1-2", "green"),
        (c3, int(stage_counts.get(3, 0)), "CKD Stade 3 (Modéré)", "orange"),
        (c4, ckd45, "CKD Stades 4-5 (Sévère)", "red"),
        (c5, high_risk, "Risque Élevé", "red"),
    ]
    for col, val, label, color in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card {color}'>
              <div class='metric-value'>{val}</div>
              <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Distribution des stades
        st.markdown("<div class='section-title'>📊 Distribution des Stades CKD</div>", unsafe_allow_html=True)
        stage_df = pd.DataFrame({
            'Stade': [STAGE_NAMES.get(int(s), f'CKD {s}') for s in stage_counts.index],
            'Nombre': stage_counts.values,
            'Stade_num': stage_counts.index.astype(int)
        })
        fig_stage = px.bar(
            stage_df, x='Stade', y='Nombre',
            color='Stade_num',
            color_continuous_scale=['#2e7d32', '#1565c0', '#f9a825', '#e65100', '#c62828'],
            text='Nombre',
        )
        fig_stage.update_traces(textposition='outside', textfont_size=14)
        fig_stage.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=20, b=20), height=320,
            xaxis_title='', yaxis_title='Nombre de patients',
            font=dict(family='Manrope')
        )
        fig_stage.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_stage, use_container_width=True)

    with col_right:
        # Répartition par risque
        st.markdown("<div class='section-title'>⚠️ Catégories de Risque</div>", unsafe_allow_html=True)
        risk_counts = df['risk_category'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
            hole=0.55,
        )
        fig_risk.update_traces(textposition='outside', textinfo='percent+label',
                               textfont_size=13)
        fig_risk.update_layout(
            showlegend=True, height=320,
            margin=dict(t=20, b=20),
            font=dict(family='Manrope')
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Ligne 2 ───────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>👥 Répartition Sexe × Stade CKD</div>", unsafe_allow_html=True)
        sex_stage = df.dropna(subset=['stage', 'sexe']).groupby(['sexe', 'stage']).size().reset_index(name='count')
        sex_stage['stage_label'] = sex_stage['stage'].apply(lambda x: f'CKD {int(x)}')
        fig_sex = px.bar(
            sex_stage, x='stage_label', y='count', color='sexe',
            barmode='group',
            color_discrete_map={'F': '#ec407a', 'M': '#42a5f5'},
            labels={'count': 'Nombre', 'stage_label': 'Stade', 'sexe': 'Sexe'}
        )
        fig_sex.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              height=280, margin=dict(t=10, b=10),
                              font=dict(family='Manrope'))
        st.plotly_chart(fig_sex, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>🎂 Distribution de l'Âge par Stade</div>", unsafe_allow_html=True)
        df_age = df.dropna(subset=['age', 'stage'])
        df_age = df_age[df_age['age'] > 0]
        df_age['stage_label'] = df_age['stage'].apply(lambda x: f'CKD {int(x)}')
        fig_age = px.box(
            df_age, x='stage_label', y='age',
            color='stage_label',
            color_discrete_map={f'CKD {k}': v for k, v in STAGE_COLORS.items()},
            labels={'age': 'Âge (ans)', 'stage_label': 'Stade'}
        )
        fig_age.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                              height=280, margin=dict(t=10, b=10),
                              font=dict(family='Manrope'))
        st.plotly_chart(fig_age, use_container_width=True)

    # ── Facteurs de risque ────────────────────────────────
    st.markdown("<div class='section-title'>🔍 Prévalence des Facteurs de Risque</div>", unsafe_allow_html=True)
    risk_factors = {
        'HTA': df.get('atcd_hta', pd.Series([0]*len(df))).sum(),
        'Diabète': df.get('atcd_diabete', pd.Series([0]*len(df))).sum(),
        'IRC Antécédent': df.get('atcd_irc', pd.Series([0]*len(df))).sum(),
        'Cardiopathie': df.get('atcd_cardio', pd.Series([0]*len(df))).sum(),
        'Tabac': df.get('tabac', pd.Series([0]*len(df))).sum(),
        'Alcool': df.get('alcool', pd.Series([0]*len(df))).sum(),
        'Phytothérapie': df.get('phytotherapie', pd.Series([0]*len(df))).sum(),
        'ATCD Fam. HTA': df.get('fam_hta', pd.Series([0]*len(df))).sum(),
    }
    rf_df = pd.DataFrame({
        'Facteur': list(risk_factors.keys()),
        'Patients': list(risk_factors.values()),
        'Pct': [v/total*100 for v in risk_factors.values()]
    }).sort_values('Patients', ascending=True)

    fig_rf = px.bar(rf_df, x='Patients', y='Facteur', orientation='h',
                    text=rf_df['Pct'].apply(lambda x: f'{x:.1f}%'),
                    color='Patients', color_continuous_scale='Blues')
    fig_rf.update_traces(textposition='outside')
    fig_rf.update_layout(showlegend=False, coloraxis_showscale=False,
                         plot_bgcolor='white', paper_bgcolor='white',
                         height=320, margin=dict(t=10, b=10, r=80),
                         font=dict(family='Manrope'))
    st.plotly_chart(fig_rf, use_container_width=True)


# ─────────────────────────────────────────────────────────
# ② ANALYSE EXPLORATOIRE
# ─────────────────────────────────────────────────────────
elif page == "📊 Analyse Exploratoire":
    page_hero(
        "📊 Analyse Exploratoire des Données",
        "Explorer les biomarqueurs et les corrélations avec une lecture clinique plus élégante.",
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Biomarqueurs", "🔗 Corrélations", "📋 Stats descriptives", "🩺 Comorbidités"
    ])

    with tab1:
        st.markdown("#### Distribution des biomarqueurs clés par stade CKD")
        biomarker = st.selectbox("Choisir un biomarqueur", [
            ('eGFR', 'DFG estimé (mL/min/1.73m²)'),
            ('creatinine', 'Créatinine (mg/L)'),
            ('uree', 'Urée (g/L)'),
            ('hb', 'Hémoglobine (g/dL)'),
            ('systole', 'Tension systolique (mmHg)'),
            ('k', 'Potassium K⁺ (meq/L)'),
            ('na', 'Sodium Na⁺ (meq/L)'),
            ('glycemie', 'Glycémie à jeun (g/L)'),
        ], format_func=lambda x: x[1])

        col_name, col_label = biomarker
        dft = df.dropna(subset=[col_name, 'stage'])
        dft['stage_label'] = dft['stage'].apply(lambda x: f'CKD {int(x)}')

        col_v, col_b = st.columns(2)
        with col_v:
            fig_v = px.violin(
                dft, x='stage_label', y=col_name,
                color='stage_label',
                color_discrete_map={f'CKD {k}': v for k, v in STAGE_COLORS.items()},
                box=True, points='outliers',
                labels={col_name: col_label, 'stage_label': 'Stade CKD'}
            )
            fig_v.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                                height=380, margin=dict(t=30, b=10),
                                title=f'Distribution : {col_label}',
                                font=dict(family='Manrope'))
            st.plotly_chart(fig_v, use_container_width=True)

        with col_b:
            # Scatterplot eGFR vs Creatinine
            dfs = df.dropna(subset=['eGFR', 'creatinine', 'stage'])
            dfs['stage_label'] = dfs['stage'].apply(lambda x: f'CKD {int(x)}')
            fig_s = px.scatter(
                dfs, x='creatinine', y='eGFR',
                color='stage_label',
                color_discrete_map={f'CKD {k}': v for k, v in STAGE_COLORS.items()},
                hover_data=['age'],
                labels={'creatinine': 'Créatinine (mg/L)', 'eGFR': 'eGFR (mL/min/1.73m²)',
                        'stage_label': 'Stade'},
                title='Créatinine vs eGFR',
                opacity=0.75, size_max=8
            )
            fig_s.update_layout(plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='white',
                                height=380, margin=dict(t=40, b=10),
                                font=dict(family='Manrope'))
            st.plotly_chart(fig_s, use_container_width=True)

    with tab2:
        st.markdown("#### Matrice de corrélation des biomarqueurs")
        num_cols = ['age', 'eGFR', 'creatinine', 'uree', 'hb', 'systole', 'diastole',
                    'na', 'k', 'glycemie', 'hematocrite', 'stage']
        available = [c for c in num_cols if c in df.columns]
        labels_map = {c: FEATURE_LABELS.get(c, c) for c in available}

        corr_df = df[available].dropna(thresh=len(available)//2)
        corr = corr_df.corr(numeric_only=True)
        corr_labels = [labels_map[c] for c in corr.columns]

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr_labels, y=corr_labels,
            colorscale='RdBu', zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}', textfont_size=9,
            hoverongaps=False
        ))
        fig_corr.update_layout(
            title='Corrélations entre variables numériques',
            height=560, width=800,
            margin=dict(t=50, b=10, l=10, r=10),
            font=dict(family='Manrope')
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("#### Statistiques descriptives des variables numériques")
        num_vars = ['age', 'eGFR', 'creatinine', 'uree', 'hb', 'systole', 'diastole',
                    'glycemie', 'na', 'k', 'risk_score']
        available_v = [c for c in num_vars if c in df.columns]
        desc = df[available_v].describe().T
        desc.index = [FEATURE_LABELS.get(c, c) for c in desc.index]
        desc = desc.round(2)
        st.dataframe(desc, use_container_width=True)

        # Valeurs manquantes
        st.markdown("#### Taux de complétude des données")
        miss_data = []
        for c in available_v:
            n_miss = df[c].isna().sum()
            miss_data.append({'Variable': FEATURE_LABELS.get(c, c),
                               'Présentes': len(df) - n_miss,
                               'Manquantes': n_miss,
                               'Complétude (%)': round((1 - n_miss/len(df))*100, 1)})
        miss_df = pd.DataFrame(miss_data).sort_values('Complétude (%)', ascending=True)
        fig_miss = px.bar(miss_df, x='Complétude (%)', y='Variable', orientation='h',
                          color='Complétude (%)',
                          color_continuous_scale=['#c62828', '#f9a825', '#2e7d32'],
                          range_color=[0, 100],
                          text='Complétude (%)')
        fig_miss.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_miss.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                               height=420, margin=dict(t=10, b=10, r=80),
                               font=dict(family='Manrope'))
        st.plotly_chart(fig_miss, use_container_width=True)

    with tab4:
        st.markdown("#### Comorbidités et facteurs de risque par stade CKD")
        comorbidities = {
            'HTA': 'atcd_hta', 'Diabète': 'atcd_diabete',
            'IRC': 'atcd_irc', 'Cardiopathie': 'atcd_cardio',
            'Tabac': 'tabac', 'Phytothérapie': 'phytotherapie',
            'OMI': 'symp_omi', 'Anémie': 'symp_anemie',
        }
        comor_rows = []
        for stage_val in sorted(df['stage'].dropna().unique()):
            subset = df[df['stage'] == stage_val]
            row = {'Stade': f'CKD {int(stage_val)}', 'N': len(subset)}
            for name, col in comorbidities.items():
                if col in df.columns:
                    row[name] = round(subset[col].fillna(0).mean() * 100, 1)
            comor_rows.append(row)
        comor_df = pd.DataFrame(comor_rows)

        # Heatmap comorbidités
        heat_cols = [c for c in comorbidities.keys() if c in comor_df.columns]
        fig_ch = px.imshow(
            comor_df[heat_cols].values,
            x=heat_cols, y=comor_df['Stade'],
            color_continuous_scale='YlOrRd',
            text_auto=True,
            labels=dict(color='Prévalence (%)'),
            title='Prévalence des comorbidités par stade (%)'
        )
        fig_ch.update_layout(height=380, font=dict(family='Manrope'))
        st.plotly_chart(fig_ch, use_container_width=True)


# ─────────────────────────────────────────────────────────
# ③ CARTOGRAPHIE
# ─────────────────────────────────────────────────────────
elif page == "🗺️ Cartographie des Risques":
    page_hero(
        "🗺️ Cartographie des Risques",
        "Visualisation géographique des niveaux de risque et de sévérité CKD au Bénin.",
    )

    if 'departement' in df.columns:
        dept_stats = df.dropna(subset=['departement']).groupby('departement').agg(
            n_patients=('stage', 'count'),
            mean_stage=('stage', 'mean'),
            mean_risk=('risk_score', 'mean'),
            n_ckd45=('stage', lambda x: (x >= 4).sum()),
            n_high_risk=('risk_category', lambda x: (x == 'Élevé').sum())
        ).reset_index()
        dept_stats['pct_severe'] = (dept_stats['n_ckd45'] / dept_stats['n_patients'] * 100).round(1)

        # Enrichir avec coordonnées
        dept_stats['lat'] = dept_stats['departement'].map(lambda d: BENIN_GEO.get(d, (9, 2.2))[0])
        dept_stats['lon'] = dept_stats['departement'].map(lambda d: BENIN_GEO.get(d, (9, 2.2))[1])

        st.markdown("##### Filtres")
        fc1, fc2 = st.columns(2)
        with fc1:
            map_metric = st.selectbox("Indicateur à visualiser", [
                ('mean_risk', '⚠️ Score de risque moyen'),
                ('mean_stage', '📊 Stade CKD moyen'),
                ('pct_severe', '🔴 % CKD Stades 4-5'),
                ('n_patients', '👥 Nombre de patients'),
            ], format_func=lambda x: x[1])
        with fc2:
            min_patients = st.slider("Patients minimum par département", 1, 20, 2)

        dept_filtered = dept_stats[dept_stats['n_patients'] >= min_patients]
        metric_col, metric_label = map_metric

        # Carte Plotly
        fig_map = px.scatter_mapbox(
            dept_filtered,
            lat='lat', lon='lon',
            size='n_patients',
            color=metric_col,
            color_continuous_scale='YlOrRd',
            hover_name='departement',
            hover_data={
                'n_patients': True, 'mean_stage': ':.2f',
                'mean_risk': ':.1f', 'pct_severe': ':.1f',
                'lat': False, 'lon': False
            },
            size_max=45,
            zoom=6,
            center={"lat": 9.3, "lon": 2.3},
            mapbox_style='carto-positron',
            title=f'Bénin — {metric_label}',
            labels={metric_col: metric_label}
        )
        fig_map.update_layout(
            height=560, margin=dict(t=40, b=0, l=0, r=0),
            coloraxis_colorbar=dict(title=metric_label, len=0.7),
            font=dict(family='Manrope')
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Tableau récapitulatif
        st.markdown("##### Résumé par département")
        display_df = dept_stats[['departement', 'n_patients', 'mean_stage',
                                  'mean_risk', 'n_ckd45', 'pct_severe']].copy()
        display_df.columns = ['Département', 'Patients', 'Stade Moyen',
                               'Score Risque Moyen', 'CKD 4-5', '% CKD Sévère']
        display_df = display_df.sort_values('Score Risque Moyen', ascending=False)
        display_df['Stade Moyen'] = display_df['Stade Moyen'].round(2)
        display_df['Score Risque Moyen'] = display_df['Score Risque Moyen'].round(1)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig_bar = px.bar(
            display_df.head(10), x='Département', y='Score Risque Moyen',
            color='Score Risque Moyen',
            color_continuous_scale='YlOrRd',
            text='Score Risque Moyen',
            title='Top 10 Départements — Score de Risque Moyen'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, coloraxis_showscale=False,
                               plot_bgcolor='white', paper_bgcolor='white',
                               height=350, margin=dict(t=40, b=10),
                               font=dict(family='Manrope'))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("La colonne 'departement' n'est pas disponible dans les données.")


# ─────────────────────────────────────────────────────────
# ④ PRÉDICTION IA
# ─────────────────────────────────────────────────────────
elif page == "🤖 Prédiction IA":
    page_hero(
        "🤖 Prédiction IA en Parcours Guidé",
        "Un workflow clinique en 5 pages pour saisir les informations patient proprement.",
    )

    if 'best_model' not in models:
        st.error("⚠️ Modèle non chargé. Lancez d'abord l'entraînement : `python models/train_model.py`")
        st.stop()

    best_model = models['best_model']
    feature_cols = meta.get('feature_cols', [c for c in FEATURE_COLS if c in df.columns])
    steps = [
        "1. Infos de base",
        "2. Biomarqueurs",
        "3. Hémodynamique",
        "4. Antécédents",
        "5. Symptômes & synthèse",
    ]

    if "pred_step" not in st.session_state:
        st.session_state.pred_step = 1

    def safe_rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    step = st.session_state.pred_step
    progress = (step - 1) / (len(steps) - 1)
    st.progress(progress)

    chips_html = []
    for i, label in enumerate(steps, start=1):
        chip_class = "step-chip"
        if i < step:
            chip_class += " done"
        elif i == step:
            chip_class += " active"
        chips_html.append(f"<div class='{chip_class}'>{label}</div>")
    st.markdown(f"<div class='step-row'>{''.join(chips_html)}</div>", unsafe_allow_html=True)

    st.markdown("<div class='wizard-wrap'>", unsafe_allow_html=True)
    if step == 1:
        st.subheader("👤 Informations générales")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Âge (ans)", 1, 100, key="pred_age", value=50)
        with c2:
            st.selectbox("Sexe", ["F", "M"], key="pred_sex")
        with c3:
            st.selectbox("Département", list(BENIN_GEO.keys()), key="pred_dept")

    elif step == 2:
        st.subheader("🧪 Biomarqueurs biologiques")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.number_input("Créatinine (mg/L)", 0.5, 3000.0, key="pred_creat", value=100.0, step=1.0)
            st.number_input("Urée (g/L)", 0.01, 30.0, key="pred_uree", value=0.8, step=0.01, format="%.2f")
        with c2:
            st.number_input("Hémoglobine (g/dL)", 1.0, 20.0, key="pred_hb", value=12.0, step=0.1, format="%.1f")
            st.number_input("Hématocrite (%)", 5.0, 65.0, key="pred_hte", value=36.0, step=0.5)
        with c3:
            st.number_input("Sodium Na⁺ (meq/L)", 100.0, 170.0, key="pred_na", value=138.0, step=0.5)
            st.number_input("Potassium K⁺ (meq/L)", 1.0, 10.0, key="pred_k", value=4.5, step=0.1, format="%.1f")
        with c4:
            st.number_input("Glycémie à jeun (g/L)", 0.1, 10.0, key="pred_gly", value=1.0, step=0.05, format="%.2f")
            st.select_slider("Protéinurie", options=[0, 0.5, 1, 2, 3, 4], key="pred_prot", value=0)

    elif step == 3:
        st.subheader("💓 Paramètres hémodynamiques")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Tension systolique (mmHg)", 70, 280, key="pred_sys", value=130)
            st.number_input("Tension diastolique (mmHg)", 40, 180, key="pred_dia", value=80)
        with c2:
            st.number_input("Pouls (bpm)", 30, 200, key="pred_pouls", value=80)
        with c3:
            st.number_input("Calcium Ca²⁺ (meq/L)", 1.0, 15.0, key="pred_ca", value=9.0, step=0.1)

    elif step == 4:
        st.subheader("🏥 Antécédents et facteurs de risque")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.checkbox("HTA connue", key="pred_atcd_hta")
            st.checkbox("Diabète", key="pred_atcd_diab")
        with c2:
            st.checkbox("IRC antérieure", key="pred_atcd_irc")
            st.checkbox("Cardiopathie/AVC", key="pred_atcd_cardio")
        with c3:
            st.checkbox("Tabac", key="pred_tabac")
            st.checkbox("Alcool", key="pred_alcool")
        with c4:
            st.checkbox("Phytothérapie trad.", key="pred_phyto")
            st.checkbox("ATCD fam. HTA", key="pred_fam_hta")

    else:
        st.subheader("🤒 Symptômes, bandelette et synthèse")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.checkbox("HTA symptomatique", key="pred_s_hta")
            st.checkbox("Anémie", key="pred_s_anemie")
        with c2:
            st.checkbox("Oligurie", key="pred_s_oligurie")
            st.checkbox("Œdème membres inf.", key="pred_s_omi")
        with c3:
            st.checkbox("BU: Hématurie", key="pred_bu_hema")
            st.checkbox("BU: Glucosurie", key="pred_bu_gluco")
        with c4:
            st.checkbox("ATCD fam. Diabète", key="pred_fam_diab")

        egfr_preview = eGFR_CKD_EPI(
            st.session_state.get("pred_creat", 100.0),
            st.session_state.get("pred_age", 50),
            st.session_state.get("pred_sex", "F"),
        )
        preview_txt = f"{egfr_preview:.1f}" if egfr_preview else "N/D"
        st.caption(f"Aperçu eGFR estimé (avant prédiction): {preview_txt} mL/min/1.73m²")
    st.markdown("</div>", unsafe_allow_html=True)

    nav_left, nav_mid, nav_right = st.columns([1, 1, 2])
    with nav_left:
        if st.button("⬅️ Précédent", disabled=(step == 1), use_container_width=True):
            st.session_state.pred_step = max(1, step - 1)
            safe_rerun()
    with nav_mid:
        if st.button("Suivant ➡️", disabled=(step == len(steps)), use_container_width=True):
            st.session_state.pred_step = min(len(steps), step + 1)
            safe_rerun()
    with nav_right:
        run_pred = st.button(
            "🔮 Lancer la prédiction",
            use_container_width=True,
            type="primary",
            disabled=(step != len(steps)),
        )

    if run_pred:
        age = st.session_state.get("pred_age", 50)
        sex = st.session_state.get("pred_sex", "F")
        creat = st.session_state.get("pred_creat", 100.0)
        egfr_val = eGFR_CKD_EPI(creat, age, sex)

        patient_data = {
            'age': age, 'sexe_bin': 1 if sex == 'M' else 0,
            'creatinine': creat, 'uree': st.session_state.get("pred_uree", 0.8),
            'eGFR': egfr_val if egfr_val else 50,
            'systole': st.session_state.get("pred_sys", 130),
            'diastole': st.session_state.get("pred_dia", 80),
            'hb': st.session_state.get("pred_hb", 12.0),
            'hematocrite': st.session_state.get("pred_hte", 36.0),
            'glycemie': st.session_state.get("pred_gly", 1.0),
            'na': st.session_state.get("pred_na", 138.0),
            'k': st.session_state.get("pred_k", 4.5),
            'ca': st.session_state.get("pred_ca", 9.0),
            'pouls': st.session_state.get("pred_pouls", 80),
            'proteinurie_score': st.session_state.get("pred_prot", 0),
            'atcd_hta': int(st.session_state.get("pred_atcd_hta", False)),
            'atcd_diabete': int(st.session_state.get("pred_atcd_diab", False)),
            'atcd_irc': int(st.session_state.get("pred_atcd_irc", False)),
            'atcd_cardio': int(st.session_state.get("pred_atcd_cardio", False)),
            'tabac': int(st.session_state.get("pred_tabac", False)),
            'alcool': int(st.session_state.get("pred_alcool", False)),
            'phytotherapie': int(st.session_state.get("pred_phyto", False)),
            'symp_hta': int(st.session_state.get("pred_s_hta", False)),
            'symp_anemie': int(st.session_state.get("pred_s_anemie", False)),
            'symp_oligurie': int(st.session_state.get("pred_s_oligurie", False)),
            'symp_omi': int(st.session_state.get("pred_s_omi", False)),
            'fam_hta': int(st.session_state.get("pred_fam_hta", False)),
            'fam_diabete': int(st.session_state.get("pred_fam_diab", False)),
            'bu_hematurie': int(st.session_state.get("pred_bu_hema", False)),
            'bu_glucosurie': int(st.session_state.get("pred_bu_gluco", False)),
        }

        X_patient = pd.DataFrame([patient_data])
        for col in feature_cols:
            if col not in X_patient.columns:
                X_patient[col] = 0
        X_patient = X_patient[feature_cols]

        try:
            pred_stage = best_model.predict(X_patient)[0]
            pred_proba = best_model.predict_proba(X_patient)[0]
            classes = best_model.classes_
        except Exception as e:
            st.error(
                "Erreur de compatibilité modèle/environnement pendant la prédiction. "
                "Veuillez réentraîner et republier les modèles avec les mêmes versions de dépendances."
            )
            st.exception(e)
            st.stop()

        tmp = pd.DataFrame([patient_data])
        for col in ['atcd_hta', 'atcd_diabete', 'atcd_cardio', 'tabac', 'alcool', 'phytotherapie']:
            if col not in tmp.columns:
                tmp[col] = 0
        risk_val = float(compute_risk_score(tmp).iloc[0])
        risk_cat = 'Faible' if risk_val < 33 else ('Modéré' if risk_val < 66 else 'Élevé')

        st.markdown("---")
        st.markdown("## 📋 Résultats de la Prédiction")
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            stage_color = STAGE_COLORS.get(pred_stage, '#666')
            st.markdown(f"""
            <div style='background:{stage_color}15; border:2px solid {stage_color}; border-radius:16px; padding:1.5rem; text-align:center;'>
              <div style='font-size:3rem;'>{'🟢' if pred_stage <= 2 else '🟡' if pred_stage == 3 else '🔴'}</div>
              <div style='font-size:1.8rem; font-weight:700; color:{stage_color}; margin:0.5rem 0;'>CKD Stade {pred_stage}</div>
                            <div style='color:#000000; font-size:0.9rem; font-weight:600;'>{STAGE_NAMES.get(pred_stage, '')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            risk_color = RISK_COLORS.get(risk_cat, '#666')
            st.markdown(f"""
            <div style='background:{risk_color}15; border:2px solid {risk_color}; border-radius:16px; padding:1.5rem; text-align:center;'>
              <div style='font-size:3rem;'>⚠️</div>
              <div style='font-size:1.8rem; font-weight:700; color:{risk_color}; margin:0.5rem 0;'>{risk_val:.1f}/100</div>
                            <div style='color:#000000; font-size:0.9rem; font-weight:600;'>Risque {risk_cat}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_res3:
            egfr_display = f"{egfr_val:.1f}" if egfr_val else "N/D"
            egfr_color = '#0f766e' if (egfr_val or 0) >= 60 else ('#f59e0b' if (egfr_val or 0) >= 30 else '#dc2626')
            st.markdown(f"""
            <div style='background:{egfr_color}15; border:2px solid {egfr_color}; border-radius:16px; padding:1.5rem; text-align:center;'>
              <div style='font-size:3rem;'>🧬</div>
              <div style='font-size:1.8rem; font-weight:700; color:{egfr_color}; margin:0.5rem 0;'>{egfr_display}</div>
                            <div style='color:#000000; font-size:0.9rem; font-weight:600;'>eGFR mL/min/1.73m²</div>
            </div>
            """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_val,
            title={'text': "Score de Risque Global", 'font': {'size': 18, 'family': 'Manrope'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 12}},
                'bar': {'color': RISK_COLORS.get(risk_cat, '#0077b6'), 'thickness': 0.3},
                'steps': [
                    {'range': [0, 33], 'color': '#e8f5ec'},
                    {'range': [33, 66], 'color': '#fff7e7'},
                    {'range': [66, 100], 'color': '#ffecec'}
                ],
                'threshold': {'line': {'color': '#dc2626', 'width': 3}, 'thickness': 0.75, 'value': 66}
            },
            number={'suffix': '/100', 'font': {'size': 28}}
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30), font=dict(family='Manrope'))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### Probabilités par stade CKD")
        prob_df = pd.DataFrame({
            'Stade': [f'CKD {int(c)}' for c in classes],
            'Probabilité (%)': [round(p * 100, 1) for p in pred_proba],
            'stage_num': [int(c) for c in classes]
        })
        fig_prob = px.bar(
            prob_df, x='Stade', y='Probabilité (%)',
            color='stage_num',
            color_discrete_map={k: v for k, v in STAGE_COLORS.items()},
            text='Probabilité (%)',
        )
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prob.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(t=10, b=10),
            font=dict(family='Manrope')
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("#### 💊 Recommandations Cliniques")
        recs = {
            1: "✅ Fonction rénale normale ou presque normale. Surveillance annuelle recommandée. Contrôle des facteurs de risque (HTA, diabète).",
            2: "🔵 Légère réduction du DFG. Suivi tous les 6 mois. Optimiser la pression artérielle et la glycémie.",
            3: "🟡 Atteinte modérée. Référence néphrologue recommandée. Surveillance des complications (anémie, phosphocalcique). Suivi trimestriel.",
            4: "🟠 Atteinte sévère. Consultation urgente néphrologue. Préparation à la thérapie de remplacement rénal (dialyse/transplant). Suivi mensuel.",
            5: "🔴 Insuffisance rénale terminale. Initiation ou planification de la dialyse. Suivi hebdomadaire requis. Évaluation transplantation rénale.",
        }
        st.info(recs.get(pred_stage, "Consultez un néphrologue pour prise en charge."))


# ─────────────────────────────────────────────────────────
# ⑤ PERFORMANCE DES MODÈLES
# ─────────────────────────────────────────────────────────
elif page == "📈 Performance des Modèles":
    page_hero(
        "📈 Performance des Modèles",
        "Comparer la robustesse des modèles et comprendre les variables clés.",
    )

    if not meta:
        st.warning("Métadonnées des modèles non disponibles. Lancez l'entraînement.")
        st.stop()

    # ── Métriques globales ────────────────────────────────
    st.markdown("### 🏆 Comparaison des Modèles")
    col_rf, col_gb = st.columns(2)

    with col_rf:
        st.markdown(f"""
        <div style='background:#e3f2fd; border:2px solid #1565c0; border-radius:12px; padding:1.2rem; text-align:center;'>
                    <h3 style='color:#001a72; margin:0;'>🌳 Random Forest</h3>
          <div style='font-size:2.5rem; font-weight:700; color:#1a237e; margin:0.5rem 0;'>
            {meta.get('rf_cv_f1', 0):.3f}
          </div>
                    <div style='color:#000000; font-weight:600;'>F1 Score (CV 5-fold)</div>
                    <div style='color:#000000; margin-top:0.3rem; font-size:0.85rem; font-weight:500;'>
            ± {meta.get('rf_cv_std', 0):.3f} | Acc. train: {meta.get('rf_train_acc', 0):.3f}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gb:
        best = meta.get('best_model', '')
        border = '#2e7d32' if 'Gradient' in best else '#1565c0'
        st.markdown(f"""
        <div style='background:#e8f5e9; border:2px solid {border}; border-radius:12px; padding:1.2rem; text-align:center;'>
                    <h3 style='color:#001a72; margin:0;'>⚡ Gradient Boosting</h3>
          <div style='font-size:2.5rem; font-weight:700; color:#1a237e; margin:0.5rem 0;'>
            {meta.get('gb_cv_f1', 0):.3f}
          </div>
                    <div style='color:#000000; font-weight:600;'>F1 Score (CV 5-fold)</div>
                    <div style='color:#000000; margin-top:0.3rem; font-size:0.85rem; font-weight:500;'>
            ± {meta.get('gb_cv_std', 0):.3f} | Acc. train: {meta.get('gb_train_acc', 0):.3f}
          </div>
                    {'<div style="color:#001a72; font-weight:700; margin-top:0.4rem;">⭐ MEILLEUR MODÈLE</div>' if 'Gradient' in best else ''}
        </div>
        """, unsafe_allow_html=True)

    # ── Distribution des stades ───────────────────────────
    st.markdown("### 📊 Distribution des Classes d'Entraînement")
    if 'stage_distribution' in meta:
        dist = meta['stage_distribution']
        dist_df = pd.DataFrame({
            'Stade': [f'CKD {k}' for k in dist.keys()],
            'Patients': list(dist.values()),
        })
        fig_dist = px.bar(
            dist_df, x='Stade', y='Patients',
            color='Patients', color_continuous_scale='Blues',
            text='Patients',
            title='Distribution des stades dans le dataset d\'entraînement'
        )
        fig_dist.update_traces(textposition='outside')
        fig_dist.update_layout(showlegend=False, coloraxis_showscale=False,
                               plot_bgcolor='white', paper_bgcolor='white',
                               height=300, margin=dict(t=40, b=10),
                               font=dict(family='Manrope'))
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Matrice de confusion (simulation) ─────────────────
    st.markdown("### 🎯 Importance des Features (Top 15)")
    if 'importance' in meta:
        imp_df = pd.DataFrame(meta['importance']).head(15)
        imp_df = imp_df.sort_values('mean_importance')
        fig_imp = px.bar(
            imp_df, x='mean_importance', y='label', orientation='h',
            color='mean_importance', color_continuous_scale='Blues',
            labels={'mean_importance': 'Importance Moyenne', 'label': 'Feature'},
            title='Importance des variables (Random Forest + Gradient Boosting)'
        )
        fig_imp.update_layout(showlegend=False, coloraxis_showscale=False,
                              plot_bgcolor='white', paper_bgcolor='white',
                              height=480, margin=dict(t=40, b=10, r=20),
                              font=dict(family='Manrope'))
        st.plotly_chart(fig_imp, use_container_width=True)


# ─────────────────────────────────────────────────────────
# ⑥ EXPLICATIONS
# ─────────────────────────────────────────────────────────
elif page == "👁️ Explications & SHAP":
    page_hero(
        "👁️ Interprétabilité & Explications",
        "Lecture transparente des prédictions pour un usage clinique plus fiable.",
    )

    st.markdown("""
    <div style='background:linear-gradient(120deg,#eef7ff,#ecfbf4); border:1px solid #cfe3f1; border-left:5px solid #0077b6; padding:1rem 1.5rem; border-radius:12px; margin-bottom:1.2rem; color:#000000;'>
    <b>🧠 Interprétabilité du Modèle</b><br>
    Cette section présente les mécanismes d'explication du modèle : importance globale des features,
    contribution par stade, et waterfall chart d'explication individuelle (SHAP simplifié).
    </div>
    """, unsafe_allow_html=True)

    tab_global, tab_stage, tab_individual = st.tabs([
        "🌍 Importance Globale", "📊 Par Stade CKD", "👤 Patient Individuel"
    ])

    with tab_global:
        if 'importance' in meta:
            imp_df = pd.DataFrame(meta['importance'])

            # Treemap
            fig_tree = px.treemap(
                imp_df.head(20),
                values='mean_importance', names='label',
                color='mean_importance', color_continuous_scale='Blues',
                title='Carte d\'importance des variables (Treemap)'
            )
            fig_tree.update_layout(height=420, font=dict(family='Manrope'),
                                   margin=dict(t=40, b=10))
            st.plotly_chart(fig_tree, use_container_width=True)

            # Radar chart des catégories
            categories = {
                'Biomarqueurs Rénaux': ['eGFR', 'creatinine', 'uree'],
                'Paramètres Sanguins': ['hb', 'hematocrite', 'na', 'k', 'ca'],
                'Hémodynamique': ['systole', 'diastole', 'pouls'],
                'Antécédents': ['atcd_hta', 'atcd_diabete', 'atcd_irc', 'atcd_cardio'],
                'Mode de Vie': ['tabac', 'alcool', 'phytotherapie'],
                'Démographie': ['age', 'sexe_bin'],
            }
            cat_scores = {}
            for cat, feats in categories.items():
                feats_in = [f for f in feats if any(r['feature'] == f for _, r in imp_df.iterrows())]
                if feats_in:
                    s = imp_df[imp_df['feature'].isin(feats_in)]['mean_importance'].sum()
                    cat_scores[cat] = round(s * 100, 2)

            if cat_scores:
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=list(cat_scores.values()),
                    theta=list(cat_scores.keys()),
                    fill='toself',
                    fillcolor='rgba(21,101,192,0.2)',
                    line_color='#1565c0',
                    name='Importance'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title='Importance par catégorie de variables',
                    height=420, font=dict(family='Manrope'),
                    margin=dict(t=50, b=10)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    with tab_stage:
        st.markdown("#### Facteurs de risque prévalents par stade CKD")
        risk_factors_by_stage = []
        for stage_val in sorted(df['stage'].dropna().unique()):
            subset = df[df['stage'] == stage_val]
            factors = {
                'eGFR moyen': subset['eGFR'].mean(),
                'Créatinine moy.': subset['creatinine'].mean(),
                'Hb moyenne': subset['hb'].mean(),
                'TA systolique': subset['systole'].mean(),
                '% HTA': subset.get('atcd_hta', pd.Series([0]*len(subset))).mean()*100,
                '% Diabète': subset.get('atcd_diabete', pd.Series([0]*len(subset))).mean()*100,
            }
            factors['Stade'] = f'CKD {int(stage_val)}'
            risk_factors_by_stage.append(factors)
        stage_df = pd.DataFrame(risk_factors_by_stage).set_index('Stade')

        for col in ['eGFR moyen', 'Créatinine moy.', 'Hb moyenne', 'TA systolique']:
            if col in stage_df.columns:
                fig_line = px.line(
                    stage_df.reset_index(), x='Stade', y=col,
                    markers=True, title=f'{col} par stade CKD',
                    color_discrete_sequence=['#1565c0']
                )
                fig_line.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                       height=250, margin=dict(t=40, b=10),
                                       font=dict(family='Manrope'))
                st.plotly_chart(fig_line, use_container_width=True)

    with tab_individual:
        st.markdown("#### Explication SHAP simplifiée pour un patient du dataset")
        idx = st.slider("Sélectionner un patient (index)", 0, len(df)-1, 0)
        patient_row = df.iloc[idx]

        if 'best_model' in models:
            X_all, y_all, feat_cols = get_feature_matrix(df)
            if idx in X_all.index:
                X_pt = X_all.loc[[idx]]
                pred = models['best_model'].predict(X_pt)[0]
                proba = models['best_model'].predict_proba(X_pt)[0]
                classes = models['best_model'].classes_

                st.markdown(f"""
                **Patient #{idx+1}** | Stade réel: `CKD {int(patient_row.get('stage', '?'))}` 
                | Stade prédit: `CKD {pred}`
                | Département: `{patient_row.get('departement', 'N/D')}`
                | eGFR: `{patient_row.get('eGFR', 'N/D'):.1f} mL/min` 
                """)

                # Feature contributions approximées
                avg_values = X_all.mean()
                pt_values = X_pt.iloc[0]
                contributions = (pt_values - avg_values).abs()

                if hasattr(models['best_model'].named_steps['clf'], 'feature_importances_'):
                    fi = models['best_model'].named_steps['clf'].feature_importances_
                    shap_approx = (pt_values.values - avg_values.values) * fi
                    contrib_df = pd.DataFrame({
                        'Feature': [FEATURE_LABELS.get(c, c) for c in feat_cols],
                        'Contribution': shap_approx,
                    }).sort_values('Contribution')

                    # Waterfall-like chart
                    contrib_df['Color'] = contrib_df['Contribution'].apply(
                        lambda x: '#c62828' if x > 0 else '#2e7d32'
                    )
                    fig_wf = go.Figure(go.Bar(
                        x=contrib_df['Contribution'].tail(15),
                        y=contrib_df['Feature'].tail(15),
                        orientation='h',
                        marker_color=contrib_df['Color'].tail(15),
                    ))
                    fig_wf.update_layout(
                        title=f'Contribution des features — Patient #{idx+1}',
                        xaxis_title='Contribution au score (rouge = risque ↑, vert = risque ↓)',
                        plot_bgcolor='white', paper_bgcolor='white',
                        height=480, margin=dict(t=50, b=10, l=10),
                        font=dict(family='Manrope')
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

                # Probabilités
                prob_df = pd.DataFrame({
                    'Stade': [f'CKD {int(c)}' for c in classes],
                    'Proba (%)': [round(p*100, 1) for p in proba],
                    'c': [int(c) for c in classes]
                })
                fig_p = px.bar(prob_df, x='Stade', y='Proba (%)',
                               color='c', color_discrete_map=STAGE_COLORS,
                               text='Proba (%)', title='Probabilités de chaque stade')
                fig_p.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_p.update_layout(showlegend=False, coloraxis_showscale=False,
                                    plot_bgcolor='white', paper_bgcolor='white',
                                    height=280, margin=dict(t=40, b=10),
                                    font=dict(family='Manrope'))
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.warning("Patient non disponible dans la matrice de features.")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#000000; font-size:0.8rem; padding: 0.6rem 0.5rem 0.8rem;'>
  🫀 <b style='color:#023e8a;'>Olanminché</b> — Plateforme IA de détection précoce de la Maladie Rénale Chronique<br>
  Expérience clinique interactive · Bénin · 2026
</div>
""", unsafe_allow_html=True)
