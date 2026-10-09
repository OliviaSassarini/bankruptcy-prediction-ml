"""
train.py - BUILD, COMPARE AND EVALUATE THE MODELS
=================================================
What this file does:
    1. get_models(): sets up 3 models - Logistic Regression (simple and
       explainable), Random Forest and Gradient Boosting (more powerful).
       All use "class_weight=balanced" because bankruptcies are rare.
    2. best_threshold(): instead of the default 50% cut-off, finds the
       probability cut-off that gives the lowest business cost
       (missed bankruptcy = 10, false alarm = 1, set in config.py).
       It is chosen using cross-validation on the TRAINING data only,
       so the test set stays untouched for a fair final score.
    3. evaluate(): scores each model on the test set - ROC-AUC, PR-AUC
       (better for rare events), recall, precision and business cost.
    4. Charts: ROC and precision-recall curves, a confusion matrix and the
       most important ratios, saved to figures/.
    5. fit_scorecard(): a small logistic model using only the 9 key ratios,
       whose coefficients are exported to Excel.
    6. run_training(): runs everything, picks the model with the lowest
       business cost, and saves it to models/best_model.joblib.

Run by: main.py.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import config


def get_models():
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000, random_state=rs),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=3,
            n_jobs=-1, random_state=rs,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.05, max_iter=300, random_state=rs,
        ),
    }


def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * config.COST_MISSED_BANKRUPTCY + fp * config.COST_FALSE_ALARM


def best_threshold(y_true, proba):
    """Pick the probability cut-off that minimises business cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [business_cost(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


def evaluate(name, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "threshold": round(threshold, 2),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "recall_bankrupt": recall_score(y_true, pred, zero_division=0),
        "precision_bankrupt": precision_score(y_true, pred, zero_division=0),
        "business_cost": business_cost(y_true, pred),
    }


def train_and_compare(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted, test_probas = [], {}, {}

    for name, model in get_models().items():
        print(f"Training {name}...")
        # Choose the threshold on out-of-fold TRAINING predictions (no test leakage)
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = best_threshold(y_train, oof)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results.append(evaluate(name, y_test, proba, threshold))
        fitted[name], test_probas[name] = model, proba

    results = pd.DataFrame(results).sort_values("business_cost").reset_index(drop=True)
    print("\n", results.round(3).to_string(index=False))
    return results, fitted, test_probas


def plot_curves(y_test, test_probas):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.2f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax2.plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, proba):.2f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=0.8, label="Random guess")
    ax2.set(title="Precision-Recall curve (better for rare events)", xlabel="Recall", ylabel="Precision")
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "04_roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_test, proba, threshold, name):
    pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Healthy", "Bankrupt"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{name} @ threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "05_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, name, n=15):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "named_steps"):
        imp = pd.Series(np.abs(model[-1].coef_[0]), index=feature_names)
    else:
        return
    top = imp.sort_values().tail(n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#4C78A8")
    ax.set_title(f"Most important ratios - {name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_feature_importance.png", dpi=150)
    plt.close(fig)


def fit_scorecard(X_train, y_train):
    """A simple, explainable logistic model on the key ratios only.
    Its coefficients are exported to Excel so the scorecard works with formulas."""
    ratios = [r for r in config.KEY_RATIOS if r in X_train.columns]
    X = X_train[ratios]
    means, stds = X.mean(), X.std().replace(0, 1)
    model = LogisticRegression(class_weight="balanced", max_iter=5000)
    model.fit((X - means) / stds, y_train)
    return pd.DataFrame({
        "ratio": ratios,
        "mean": means.values,
        "std": stds.values,
        "coefficient": model.coef_[0],
    }), float(model.intercept_[0])


def run_training(X_train, X_test, y_train, y_test):
    results, fitted, test_probas = train_and_compare(X_train, X_test, y_train, y_test)
    best = results.iloc[0]
    best_name, threshold = best["model"], best["threshold"]

    plot_curves(y_test, test_probas)
    plot_confusion(y_test, test_probas[best_name], threshold, best_name)
    plot_feature_importance(fitted[best_name], X_train.columns, best_name)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best_name], "threshold": threshold, "features": list(X_train.columns)},
                config.MODELS_DIR / "best_model.joblib")
    results.to_csv(config.REPORT_DIR / "model_comparison.csv", index=False)
    with open(config.REPORT_DIR / "best_model.json", "w") as f:
        json.dump({k: (v if isinstance(v, str) else float(v)) for k, v in best.items()}, f, indent=2)

    predictions = pd.DataFrame({
        "company_id": X_test.index,
        "actual_bankrupt": y_test.values,
        "predicted_probability": test_probas[best_name],
    })
    predictions["flagged"] = (predictions["predicted_probability"] >= threshold).astype(int)

    scorecard, intercept = fit_scorecard(X_train, y_train)
    print(f"\nBest model: {best_name} (threshold {threshold:.2f})")
    return results, predictions, scorecard, intercept, threshold
