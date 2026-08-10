import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

def calculate_metrics(result):
    """
    Calculate probabilistic + classification metrics.
    """

    y_true = result[
        "Actual_Soft_Label"
    ].to_numpy()

    y_pred = result[
        "P_OPEN"
    ].to_numpy()

    # ----------------------------------------
    # Soft-label metrics
    # ----------------------------------------

    brier = np.mean(
        (y_pred - y_true) ** 2
    )

    if (
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
    ):
        correlation = np.corrcoef(
            y_true,
            y_pred
        )[0, 1]
    else:
        correlation = 0.0

    # ----------------------------------------
    # Binary reference target
    # ----------------------------------------

    y_true_binary = (
        y_true >= 0.5
    ).astype(int)

    y_pred_binary = (
        y_pred >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true_binary,
        y_pred_binary
    )

    precision = precision_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    recall = recall_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    no_rain_mask = (y_true_binary == 0)

    if no_rain_mask.sum() > 0:
        closed_accuracy = (
            y_pred_binary[no_rain_mask] == 0
        ).mean()
    else:
        closed_accuracy = 0.0

    # ----------------------------------------
    # ROC / PR
    # ----------------------------------------

    if len(
        np.unique(y_true_binary)
    ) == 2:

        roc_auc = roc_auc_score(
            y_true_binary,
            y_pred
        )

        pr_auc = average_precision_score(
            y_true_binary,
            y_pred
        )

    else:
        # No positive/negative rain event in this evaluation block
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        "Brier": brier,
        "Correlation": correlation,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Closed_Accuracy": closed_accuracy,
    }

def evaluate_fold(json_path, threshold=0.5):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    records = data["records"]

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # numeric conversion
    # --------------------------------------------------------

    numeric_cols = [
        "predicted_probability",
        "actual_future_rain_score",
        "predicted_open",
        "actual_rain",
        "hmm_state",
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "predicted_probability",
            "actual_rain",
        ]
    ).copy()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_true = (
        df["actual_rain"]
        .astype(int)
        .to_numpy()
    )

    y_prob = (
        df["predicted_probability"]
        .astype(float)
        .clip(0, 1)
        .to_numpy()
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # Specificity / Closed Accuracy
    if (tn + fp) > 0:
        closed_accuracy = tn / (tn + fp)
    else:
        closed_accuracy = np.nan

    # Rain recall
    if (tp + fn) > 0:
        rain_recall = tp / (tp + fn)
    else:
        rain_recall = np.nan

    # Open precision
    if (tp + fp) > 0:
        open_precision = tp / (tp + fp)
    else:
        open_precision = np.nan

    # F1
    if (tp + fn) > 0 and (tp + fp) > 0:
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    else:
        f1 = np.nan

    # --------------------------------------------------------
    # Probability metrics
    # --------------------------------------------------------

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    # AUC only makes sense if both classes exist
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

        pr_auc = average_precision_score(
            y_true,
            y_prob
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan

    # --------------------------------------------------------
    # Fold ID
    # --------------------------------------------------------

    fold_number = None

    try:
        fold_number = int(
            json_path.stem.split("_")[-1]
        )
    except Exception:
        pass

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if len(df) > 0:

        eval_date = (
            df["timestamp"]
            .min()
            .strftime("%Y-%m-%d")
        )

    else:

        eval_date = None

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {
        "fold": fold_number,
        "date": eval_date,

        "n_minutes": len(df),

        "rain_minutes": int(
            y_true.sum()
        ),

        "dry_minutes": int(
            (y_true == 0).sum()
        ),

        "predicted_open_minutes": int(
            y_pred.sum()
        ),

        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),

        "accuracy": accuracy,

        "closed_accuracy": closed_accuracy,

        "rain_recall": rain_recall,

        "open_precision": open_precision,

        "f1": f1,

        "brier": brier,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,
    }

def evaluate_all_folds(
    log_dir,
    n_folds=44,
    threshold=0.5
):
    results = []
    for fold in range(
        1,
        n_folds + 1
    ):

        json_path = (
            log_dir
            / f"fold_{fold:03d}.json"
        )

        if not json_path.exists():

            print(
                f"[WARNING] Missing: "
                f"{json_path}"
            )

            continue

        try:

            result = evaluate_fold(
                json_path,
                threshold
            )

            results.append(
                result
            )

            print(
                f"Fold {fold:02d} | "
                f"Rain={result['rain_minutes']:4d} | "
                f"Accuracy={result['accuracy']:.3f} | "
                f"Closed={result['closed_accuracy']:.3f} | "
                f"Recall={result['rain_recall']:.3f} | "
                f"Precision={result['open_precision']:.3f} | "
                f"F1={result['f1']:.3f}"
            )

        except Exception as e:

            print(
                f"[ERROR] Fold {fold}: "
                f"{e}"
            )

    return pd.DataFrame(results)