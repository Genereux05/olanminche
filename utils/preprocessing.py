"""
AI4CKD - Preprocessing Utilities
Nettoyage et préparation des données pour la prédiction du stade CKD
"""
import pandas as pd
import numpy as np
import re


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def parse_french_float(val):
    """Convertit un nombre au format français (virgule) en float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(',', '.').replace(' ', '')
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_creatinine(val):
    """Parse la créatinine (mg/L), corrige les outliers évidents."""
    v = parse_french_float(val)
    if v is not None and not np.isnan(v):
        if v > 5000:   # probablement µmol/L → convertir ×0.0113
            v = v * 0.0113
        if v <= 0:
            return np.nan
    return v


def parse_uree(val):
    """Parse urée (g/L). Si valeur > 50, probablement mmol/L → ×0.06."""
    v = parse_french_float(val)
    if v is not None and not np.isnan(v):
        if v > 50:
            v = v * 0.06
        if v <= 0:
            return np.nan
    return v


def parse_age(val):
    """Parse l'âge (peut être une string)."""
    v = parse_french_float(val)
    if v is not None and not np.isnan(v):
        if v < 1 or v > 120:
            return np.nan
    return v


def parse_bp(val):
    """Parse la tension artérielle."""
    v = parse_french_float(val)
    if v is not None and not np.isnan(v):
        if v < 40 or v > 300:
            return np.nan
    return v


def parse_hb(val):
    """Parse hémoglobine."""
    v = parse_french_float(val)
    if v is not None and not np.isnan(v):
        if v < 1 or v > 25:
            return np.nan
    return v


def proteinurie_encode(val):
    """Encode la protéinurie textuelle en score numérique."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    mapping = {
        'négative': 0, 'negative': 0, 'nul': 0, '0': 0,
        'traces': 0.5, 'trace': 0.5,
        '1+': 1, '+': 1,
        '2+': 2, '++': 2,
        '3+': 3, '+++': 3,
        '4+': 4, '++++': 4,
    }
    return mapping.get(val, np.nan)


def eGFR_CKD_EPI(creatinine_mg_L, age, sex):
    """
    Calcule le DFG estimé (eGFR) avec la formule CKD-EPI.
    creatinine_mg_L : créatinine en mg/L → convertir en mg/dL (/10)
    Retourne eGFR en mL/min/1.73m²
    """
    if any(pd.isna(x) for x in [creatinine_mg_L, age, sex]):
        return np.nan
    try:
        scr = creatinine_mg_L / 10.0  # mg/L → mg/dL
        if scr <= 0 or age <= 0:
            return np.nan

        if str(sex).strip().upper() in ['F', 'FEMME', 'FEMALE']:
            kappa, alpha, sex_factor = 0.7, -0.329, 1.018
        else:
            kappa, alpha, sex_factor = 0.9, -0.411, 1.0

        ratio = scr / kappa
        if ratio < 1:
            egfr = 141 * (ratio ** alpha) * (0.993 ** age) * sex_factor
        else:
            egfr = 141 * (ratio ** (-1.209)) * (0.993 ** age) * sex_factor
        return round(egfr, 2)
    except Exception:
        return np.nan


def egfr_to_ckd_stage(egfr):
    """Convertit eGFR en stade CKD (G1-G5)."""
    if pd.isna(egfr):
        return np.nan
    if egfr >= 90:
        return 1
    elif egfr >= 60:
        return 2
    elif egfr >= 45:
        return 3
    elif egfr >= 30:
        return 3
    elif egfr >= 15:
        return 4
    else:
        return 5


def stage_label(val):
    """Normalise l'étiquette de stade CKD."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper().replace(' ', '')
    mapping = {
        'CKD1': 1, 'CKD2': 2, 'CKD3': 3, 'CKD3A': 3,
        'CKD3B': 3, 'CKD4': 4, 'CKD5': 5,
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        'STADE1': 1, 'STADE2': 2, 'STADE3': 3, 'STADE4': 4, 'STADE5': 5,
    }
    return mapping.get(s, np.nan)


def stage_label_detailed(val):
    """Conserve les sous-stades 3a/3b pour affichage."""
    if pd.isna(val):
        return 'Inconnu'
    s = str(val).strip()
    return s


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Charge et nettoie le dataset AI4CKD.
    Retourne un DataFrame avec toutes les features prétraitées.
    """
    df = pd.read_csv(path, low_memory=False)

    # ── Renommer pour simplifier ──────────────────────────
    rename = {
        "Stage de l'IRC": 'stage_label',
        'Sexe': 'sexe',
        'Age': 'age',
        'Adresse (Département)': 'departement',
        'Créatinine (mg/L)': 'creatinine_raw',
        'Urée (g/L)': 'uree_raw',
        'TA (mmHg)/Systole': 'systole_raw',
        'TA (mmHg)/Diastole': 'diastole_raw',
        'Hb (g/dL)': 'hb_raw',
        'Glycémie à jeun (taux de Glucose)': 'glycemie_raw',
        'Na^+ (meq/L)': 'na_raw',
        'K^+ (meq/L)': 'k_raw',
        'Ca^2+ (meq/L)': 'ca_raw',
        'Protéinurie': 'proteinurie_txt',
        'Personnels Médicaux/HTA': 'atcd_hta',
        'Personnels Médicaux/Diabète 2': 'atcd_diabete',
        'Personnels Médicaux/IRC': 'atcd_irc',
        'Personnels Médicaux/Maladies Cardiovasculaire(Cardiopathie, AVC, preeclampsie)': 'atcd_cardio',
        'Enquête Sociale/Tabac': 'tabac',
        'Enquête Sociale/Alcool': 'alcool',
        'Enquête Sociale/Phytothérapie traditionnelle': 'phytotherapie',
        'Symptômes/HTA': 'symp_hta',
        'Symptômes/Anémie': 'symp_anemie',
        'Symptômes/Oligurie': 'symp_oligurie',
        'Symptômes/OMI': 'symp_omi',
        'Symptômes/Asthénie': 'symp_asthenie',
        'Symptômes/Nausées': 'symp_nausees',
        'Symptômes/Vomissements': 'symp_vomissements',
        'Personnels Familiaux/HTA': 'fam_hta',
        'Personnels Familiaux/Diabète': 'fam_diabete',
        'Situation Matrimoniale': 'situation_matrimoniale',
        'Profession (selon catégorie professionnelle)': 'profession',
        'Hte (%)': 'hematocrite_raw',
        'Plaquettes (g/L)': 'plaquettes_raw',
        'Poids (Kg)': 'poids_raw',
        'Taille (m)': 'taille_raw',
        'IMC': 'imc_raw',
        'BU/Protéinurie': 'bu_proteinurie',
        'BU/Hématurie': 'bu_hematurie',
        'BU/Glucosurie': 'bu_glucosurie',
        'Causes Majeure après Diagnostic/HTA': 'cause_hta',
        'Causes Majeure après Diagnostic/Diabète': 'cause_diabete',
        'Poul (bpm)': 'pouls_raw',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Parsing numérique ─────────────────────────────────
    df['age'] = df['age'].apply(parse_age)
    df['creatinine'] = df['creatinine_raw'].apply(parse_creatinine)
    df['uree'] = df['uree_raw'].apply(parse_uree)
    df['systole'] = df['systole_raw'].apply(parse_bp)
    df['diastole'] = df['diastole_raw'].apply(parse_bp)
    df['hb'] = df['hb_raw'].apply(parse_hb)
    df['glycemie'] = df['glycemie_raw'].apply(parse_french_float)
    df['na'] = df['na_raw'].apply(parse_french_float)
    df['k'] = df['k_raw'].apply(parse_french_float)
    df['ca'] = df.get('ca_raw', pd.Series([np.nan]*len(df))).apply(parse_french_float)
    df['pouls'] = df['pouls_raw'].apply(parse_french_float) if 'pouls_raw' in df.columns else np.nan
    df['hematocrite'] = df['hematocrite_raw'].apply(parse_french_float) if 'hematocrite_raw' in df.columns else np.nan
    df['poids'] = df['poids_raw'].apply(parse_french_float) if 'poids_raw' in df.columns else np.nan
    df['taille'] = df['taille_raw'].apply(parse_french_float) if 'taille_raw' in df.columns else np.nan

    # ── Protéinurie ───────────────────────────────────────
    df['proteinurie_score'] = df['proteinurie_txt'].apply(proteinurie_encode) if 'proteinurie_txt' in df.columns else np.nan

    # ── eGFR ─────────────────────────────────────────────
    df['eGFR'] = df.apply(
        lambda r: eGFR_CKD_EPI(r['creatinine'], r['age'], r.get('sexe', np.nan)),
        axis=1
    )

    # ── Stade cible ───────────────────────────────────────
    if 'stage_label' in df.columns:
        df['stage_label_full'] = df['stage_label'].apply(stage_label_detailed)
        df['stage'] = df['stage_label'].apply(stage_label)
        # Compléter les stades manquants par eGFR
        mask = df['stage'].isna() & df['eGFR'].notna()
        df.loc[mask, 'stage'] = df.loc[mask, 'eGFR'].apply(egfr_to_ckd_stage)
        df['stage'] = df['stage'].astype(float)
    else:
        df['stage'] = np.nan
        df['stage_label_full'] = 'Inconnu'

    # ── Variables binaires ────────────────────────────────
    binary_cols = [
        'atcd_hta', 'atcd_diabete', 'atcd_irc', 'atcd_cardio',
        'tabac', 'alcool', 'phytotherapie',
        'symp_hta', 'symp_anemie', 'symp_oligurie', 'symp_omi',
        'symp_asthenie', 'symp_nausees', 'symp_vomissements',
        'fam_hta', 'fam_diabete', 'cause_hta', 'cause_diabete',
        'bu_hematurie', 'bu_glucosurie',
    ]
    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # ── Sexe ──────────────────────────────────────────────
    df['sexe_bin'] = (df['sexe'].str.strip().str.upper() == 'M').astype(int) if 'sexe' in df.columns else 0

    # ── Score de risque composite ─────────────────────────
    df['risk_score'] = compute_risk_score(df)

    # ── Catégories de risque ──────────────────────────────
    df['risk_category'] = pd.cut(
        df['risk_score'],
        bins=[-np.inf, 33, 66, np.inf],
        labels=['Faible', 'Modéré', 'Élevé']
    )

    # ── Département / Géo ─────────────────────────────────
    if 'departement' in df.columns:
        df['departement'] = df['departement'].str.strip().str.title()

    return df


# ──────────────────────────────────────────────
# SCORE DE RISQUE
# ──────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Score de risque composite (0-100).
    Combine facteurs cliniques, biologiques et sociaux.
    """
    score = pd.Series(0.0, index=df.index)
    n_factors = pd.Series(0.0, index=df.index)

    # eGFR (poids fort)
    if 'eGFR' in df.columns:
        egfr_score = df['eGFR'].apply(lambda x: max(0, min(100, (1 - x/120)*100)) if not pd.isna(x) else 50)
        score += egfr_score * 3
        n_factors += 3

    # Créatinine
    if 'creatinine' in df.columns:
        cr_score = df['creatinine'].apply(lambda x: min(100, x/20*10) if not pd.isna(x) else 50)
        score += cr_score
        n_factors += 1

    # Tension systolique
    if 'systole' in df.columns:
        bp_score = df['systole'].apply(lambda x: min(100, max(0, (x-120)/80*100)) if not pd.isna(x) else 50)
        score += bp_score
        n_factors += 1

    # Hémoglobine
    if 'hb' in df.columns:
        hb_score = df['hb'].apply(lambda x: min(100, max(0, (13-x)/10*100)) if not pd.isna(x) else 50)
        score += hb_score
        n_factors += 1

    # Facteurs de risque binaires
    risk_binary = ['atcd_hta', 'atcd_diabete', 'atcd_cardio', 'tabac', 'alcool', 'phytotherapie']
    for col in risk_binary:
        if col in df.columns:
            score += df[col].fillna(0) * 15
            n_factors += 1

    # Protéinurie
    if 'proteinurie_score' in df.columns:
        score += df['proteinurie_score'].fillna(0) * 10
        n_factors += 1

    final_score = (score / n_factors.replace(0, 1)).clip(0, 100)
    return final_score.round(1)


# ──────────────────────────────────────────────
# FEATURES POUR ML
# ──────────────────────────────────────────────

FEATURE_COLS = [
    'age', 'sexe_bin', 'creatinine', 'uree', 'eGFR',
    'systole', 'diastole', 'hb', 'hematocrite',
    'glycemie', 'na', 'k', 'ca', 'pouls',
    'proteinurie_score',
    'atcd_hta', 'atcd_diabete', 'atcd_irc', 'atcd_cardio',
    'tabac', 'alcool', 'phytotherapie',
    'symp_hta', 'symp_anemie', 'symp_oligurie', 'symp_omi',
    'fam_hta', 'fam_diabete',
    'bu_hematurie', 'bu_glucosurie',
]

FEATURE_LABELS = {
    'age': 'Âge',
    'sexe_bin': 'Sexe (M=1)',
    'creatinine': 'Créatinine (mg/L)',
    'uree': 'Urée (g/L)',
    'eGFR': 'DFG estimé (mL/min/1.73m²)',
    'systole': 'Tension systolique (mmHg)',
    'diastole': 'Tension diastolique (mmHg)',
    'hb': 'Hémoglobine (g/dL)',
    'hematocrite': 'Hématocrite (%)',
    'glycemie': 'Glycémie à jeun (g/L)',
    'na': 'Sodium Na⁺ (meq/L)',
    'k': 'Potassium K⁺ (meq/L)',
    'ca': 'Calcium Ca²⁺ (meq/L)',
    'pouls': 'Pouls (bpm)',
    'proteinurie_score': 'Protéinurie (score)',
    'atcd_hta': 'ATCD: HTA',
    'atcd_diabete': 'ATCD: Diabète',
    'atcd_irc': 'ATCD: IRC',
    'atcd_cardio': 'ATCD: Cardiovasculaire',
    'tabac': 'Tabac',
    'alcool': 'Alcool',
    'phytotherapie': 'Phytothérapie trad.',
    'symp_hta': 'Symptôme: HTA',
    'symp_anemie': 'Symptôme: Anémie',
    'symp_oligurie': 'Symptôme: Oligurie',
    'symp_omi': 'Symptôme: OMI',
    'fam_hta': 'ATCD familial: HTA',
    'fam_diabete': 'ATCD familial: Diabète',
    'bu_hematurie': 'BU: Hématurie',
    'bu_glucosurie': 'BU: Glucosurie',
}


def get_feature_matrix(df: pd.DataFrame):
    """Retourne X (features) et y (stade) pour l'entraînement."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()

    # Imputation médiane
    for col in X.columns:
        median = X[col].median()
        X[col] = X[col].fillna(median if not pd.isna(median) else 0)

    y = df['stage'].dropna()
    X = X.loc[y.index]
    y = y.astype(int)
    return X, y, cols
