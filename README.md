# 🫀 Olanmiché — Dashboard IA pour la Maladie Rénale Chronique

> **Intelligence Artificielle au service de la détection précoce de la MRC**  
> Bootcamp Cohorte 1 ·  Bénin · 2025

---

## 📋 Table des Matières

1. [Présentation](#présentation)
2. [Architecture du Projet](#architecture)
3. [Installation](#installation)
4. [Lancement](#lancement)
5. [Fonctionnalités du Dashboard](#fonctionnalités)
6. [Méthodologie ML](#méthodologie)
7. [Résultats](#résultats)

---

## 🎯 Présentation

**Olanmiché** est une application web interactive de détection et de prédiction de la **Maladie Rénale Chronique (CKD)**. Elle combine :

- 🤖 **Machine Learning** (Random Forest + Gradient Boosting)
- 📊 **Analyse exploratoire** des données cliniques
- 🗺️ **Cartographie géographique** des patients au Bénin
- 👁️ **Interprétabilité** des prédictions (SHAP approché)
- 💊 **Recommandations cliniques** personnalisées

**Performance des modèles :**
- Random Forest : F1 = 0.887 (CV 5-fold)
- Gradient Boosting : F1 = 0.897 (CV 5-fold) ⭐ Meilleur modèle

---

## 🗂️ Architecture du Projet

```
ckd_project/
│
├── app.py                    # 🚀 Dashboard Streamlit principal
│
├── data/
│   └── Data_AI4CKD_Original.csv   # Dataset patients d'un hopital béninois
│
├── models/
│   ├── train_model.py        # Script d'entraînement ML
│   ├── rf_model.pkl          # Modèle Random Forest entraîné
│   ├── gb_model.pkl          # Modèle Gradient Boosting entraîné
│   ├── best_model.pkl        # Meilleur modèle (utilisé pour prédiction)
│   ├── metadata.json         # Métriques et métadonnées des modèles
│   └── classification_report.txt
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py      # Pipeline de nettoyage et feature engineering
│
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

---

## ⚙️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes

**1. Cloner ou extraire le projet**
```bash
# Si zip, extraire dans un dossier puis :
cd ckd_project
```

**2. Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv

# Activer l'environnement :
# Linux/macOS :
source venv/bin/activate

# Windows :
venv\Scripts\activate
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

---

## 🚀 Lancement

### Étape 1 — Entraîner les modèles ML
> ⚡ À faire une seule fois (les modèles sont sauvegardés en `.pkl`)

```bash
python models/train_model.py
```

Sortie attendue :
```
============================================================
  AI4CKD — Entraînement des modèles
============================================================
[1/5] Chargement et nettoyage des données...
  → 309 patients chargés
  → 306 patients avec stade connu | 30 features

[2/5] Entraînement Random Forest...
  → F1 CV (5-fold): 0.887 ± 0.028

[3/5] Entraînement Gradient Boosting...
  → F1 CV (5-fold): 0.897 ± 0.033

✅ Modèles sauvegardés dans: models/
   Meilleur modèle: Gradient Boosting (F1=0.897)
```

### Étape 2 — Lancer le Dashboard

```bash
streamlit run app.py
```

Le dashboard s'ouvre automatiquement sur **http://localhost:8501**

---

## 🖥️ Fonctionnalités du Dashboard

### 🏠 Accueil & Vue d'ensemble
- KPIs clés : total patients, distribution par stade, risque élevé
- Distribution des stades CKD avec graphiques interactifs
- Répartition par sexe et âge
- Prévalence des facteurs de risque

### 📊 Analyse Exploratoire
- Violin plots des biomarqueurs par stade
- Scatter plot Créatinine vs eGFR
- Matrice de corrélation interactive
- Statistiques descriptives et taux de complétude
- Heatmap des comorbidités par stade

### 🗺️ Cartographie des Risques
- Carte interactive du Bénin (Plotly Mapbox)
- Score de risque, stade moyen, % CKD sévère par département
- Filtres dynamiques
- Tableau récapitulatif par département

### 🤖 Prédiction IA
- Formulaire de saisie patient complet (30+ variables)
- Prédiction du stade CKD en temps réel
- Score de risque composite (0-100) avec jauge visuelle
- Calcul automatique de l'eGFR (formule CKD-EPI)
- Probabilités par stade en barplot
- Recommandations cliniques personnalisées

### 📈 Performance des Modèles
- Comparaison Random Forest vs Gradient Boosting
- F1 Score, accuracy, validation croisée 5-fold
- Importance des features (top 15)

### 👁️ Explications & SHAP
- Treemap de l'importance globale des variables
- Radar chart par catégorie de variables
- Waterfall chart d'explication individuelle (SHAP approché)
- Analyse patient par patient

---

## 🧪 Méthodologie ML

### Prétraitement
- Parsing des nombres au format français (virgule → point)
- Calcul de l'eGFR via formule CKD-EPI
- Encodage de la protéinurie (texte → score 0-4)
- Imputation des valeurs manquantes (médiane)
- Correction des unités aberrantes (créatinine, urée)

### Features (30 variables)
| Catégorie | Variables |
|-----------|-----------|
| Biomarqueurs | eGFR, créatinine, urée, protéinurie |
| Hématologie | Hémoglobine, hématocrite |
| Électrolytes | Sodium, potassium, calcium |
| Hémodynamique | TA systolique/diastolique, pouls |
| Métabolisme | Glycémie à jeun |
| Antécédents | HTA, diabète, IRC, cardiopathie |
| Mode de vie | Tabac, alcool, phytothérapie |
| Symptômes | HTA, anémie, oligurie, OMI |
| Famille | ATCD HTA, ATCD diabète |
| BU | Hématurie, glucosurie |

### Modèles
- **Random Forest** : 300 arbres, max_depth=12, class_weight='balanced'
- **Gradient Boosting** : 200 estimateurs, learning_rate=0.1, max_depth=5
- Validation : StratifiedKFold (k=5)

### Score de Risque Composite
Combinaison pondérée de :
- eGFR (poids x3)
- Créatinine
- Tension artérielle
- Hémoglobine
- Facteurs de risque binaires (HTA, diabète, tabac, etc.)

---

## 📊 Résultats

| Modèle | F1 (CV) | Std | Acc. train |
|--------|---------|-----|------------|
| Random Forest | 0.887 | ±0.028 | 0.984 |
| **Gradient Boosting** | **0.897** | **±0.033** | **1.000** |

**Top 5 features les plus importantes :**
1. DFG estimé (eGFR) — 47.0%
2. Créatinine — 22.8%
3. Urée — 6.0%
4. Hémoglobine — 3.3%
5. Sodium Na⁺ — 2.7%

---

## 🏥 Contexte Médical

Les stades CKD sont définis par l'eGFR (KDIGO 2012) :

| Stade | eGFR (mL/min/1.73m²) | Description |
|-------|----------------------|-------------|
| CKD 1 | ≥ 90 | Normal / légèrement élevé |
| CKD 2 | 60–89 | Légèrement diminué |
| CKD 3a | 45–59 | Modérément diminué |
| CKD 3b | 30–44 | Modérément à sévèrement diminué |
| CKD 4 | 15–29 | Sévèrement diminué |
| CKD 5 | < 15 | Insuffisance rénale terminale |

---

## 📄 Confidentialité

Toutes les données sont utilisées uniquement dans le cadre de ce projet de recherche académique (Hackathon AI4CKD). Aucune donnée n'est transmise à des serveurs externes.

---

*Ici, c'est l'AMA !* 🎓
