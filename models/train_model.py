"""
AI4CKD - Entraînement des modèles ML
Random Forest + XGBoost pour la prédiction du stade CKD
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Ajouter le répertoire parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean, get_feature_matrix, FEATURE_COLS, FEATURE_LABELS

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'Data_AI4CKD_Original.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


def train_models(data_path=DATA_PATH, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("  AI4CKD — Entraînement des modèles")
    print("=" * 60)

    # ── Chargement ────────────────────────────────────────
    print("\n[1/5] Chargement et nettoyage des données...")
    df = load_and_clean(data_path)
    print(f"  → {len(df)} patients chargés")

    X, y, feature_cols = get_feature_matrix(df)
    print(f"  → {len(X)} patients avec stade connu | {len(feature_cols)} features")
    print(f"  → Distribution des stades:\n{pd.Series(y).value_counts().sort_index().to_string()}")

    # ── Random Forest ─────────────────────────────────────
    print("\n[2/5] Entraînement Random Forest...")
    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted')
    print(f"  → F1 CV (5-fold): {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    rf.fit(X, y)
    rf_pred = rf.predict(X)
    rf_acc = accuracy_score(y, rf_pred)
    print(f"  → Accuracy train: {rf_acc:.3f}")

    # ── Gradient Boosting ─────────────────────────────────
    print("\n[3/5] Entraînement Gradient Boosting...")
    gb = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='f1_weighted')
    print(f"  → F1 CV (5-fold): {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

    gb.fit(X, y)
    gb_pred = gb.predict(X)
    gb_acc = accuracy_score(y, gb_pred)
    print(f"  → Accuracy train: {gb_acc:.3f}")

    # ── Importance des features ───────────────────────────
    print("\n[4/5] Calcul des importances de features...")
    rf_importances = rf.named_steps['clf'].feature_importances_
    gb_importances = gb.named_steps['clf'].feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'label': [FEATURE_LABELS.get(f, f) for f in feature_cols],
        'rf_importance': rf_importances,
        'gb_importance': gb_importances,
        'mean_importance': (rf_importances + gb_importances) / 2
    }).sort_values('mean_importance', ascending=False)

    print("\n  Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['mean_importance'] * 100)
        print(f"  {row['label'][:35]:<35} {row['mean_importance']:.3f} {bar}")

    # ── Sauvegarde ────────────────────────────────────────
    print("\n[5/5] Sauvegarde des modèles...")

    with open(os.path.join(model_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(model_dir, 'gb_model.pkl'), 'wb') as f:
        pickle.dump(gb, f)

    # Choisir le meilleur modèle
    best_model = rf if rf_scores.mean() >= gb_scores.mean() else gb
    best_name = 'Random Forest' if rf_scores.mean() >= gb_scores.mean() else 'Gradient Boosting'
    with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    # Méta-données
    metadata = {
        'best_model': best_name,
        'feature_cols': feature_cols,
        'rf_cv_f1': round(float(rf_scores.mean()), 4),
        'gb_cv_f1': round(float(gb_scores.mean()), 4),
        'rf_cv_std': round(float(rf_scores.std()), 4),
        'gb_cv_std': round(float(gb_scores.std()), 4),
        'rf_train_acc': round(float(rf_acc), 4),
        'gb_train_acc': round(float(gb_acc), 4),
        'n_patients': int(len(X)),
        'stage_distribution': {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        'importance': importance_df[['feature', 'label', 'mean_importance']].to_dict('records'),
    }
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Rapport de classification
    report = classification_report(y, rf.predict(X),
                                   target_names=[f'CKD {s}' for s in sorted(y.unique())])
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Random Forest - Rapport de classification\n{'='*50}\n")
        f.write(report)

    print(f"\n✅ Modèles sauvegardés dans: {model_dir}")
    print(f"   Meilleur modèle: {best_name} (F1={max(rf_scores.mean(), gb_scores.mean()):.3f})")

    return rf, gb, best_model, metadata, importance_df, df


if __name__ == '__main__':
    train_models()
