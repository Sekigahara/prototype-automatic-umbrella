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
    Calculate probabilistic and classification metrics
    for the umbrella OPEN/CLOSE decision.
    """

    # ========================================================
    # PREDICTION AND GROUND TRUTH
    # ========================================================

    y_true = result[
        "Actual_Soft_Label"
    ].to_numpy()

    y_pred = result[
        "P_OPEN"
    ].to_numpy()

    # ========================================================
    # SOFT-LABEL METRICS
    # ========================================================

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

    # ========================================================
    # BINARY OPEN/CLOSE REFERENCE
    # ========================================================

    y_true_binary = (
        y_true >= 0.5
    ).astype(int)

    y_pred_binary = (
        y_pred >= 0.5
    ).astype(int)

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

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

    # ========================================================
    # CLOSED ACCURACY
    # ========================================================
    #
    # Among samples where the actual target is CLOSE,
    # how often does the model correctly predict CLOSE?

    closed_mask = (
        y_true_binary == 0
    )

    if closed_mask.sum() > 0:

        closed_accuracy = (
            y_pred_binary[closed_mask] == 0
        ).mean()

    else:

        closed_accuracy = 0.0

    # ========================================================
    # OPEN ACCURACY
    # ========================================================
    #
    # Among samples where the actual target is OPEN,
    # how often does the model correctly predict OPEN?
    #
    # This is effectively the same as recall.

    open_mask = (
        y_true_binary == 1
    )

    if open_mask.sum() > 0:

        open_accuracy = (
            y_pred_binary[open_mask] == 1
        ).mean()

    else:

        open_accuracy = 0.0

    # ========================================================
    # ROC / PR
    # ========================================================

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

        # Evaluation block contains only OPEN
        # or only CLOSE samples.

        roc_auc = 0.0
        pr_auc = 0.0

    # ========================================================
    # RETURN METRICS
    # ========================================================

    return {
        "Brier": brier,
        "Correlation": correlation,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Open_Accuracy": open_accuracy,
        "Closed_Accuracy": closed_accuracy,
    }

def evaluate_fold(
    json_path,
    threshold=0.5
):
    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Extract records
    # --------------------------------------------------------

    records = data["records"]

    rows = []

    for record in records:

        model = record.get(
            "model",
            {}
        )

        ground_truth = record.get(
            "ground_truth",
            {}
        )

        rows.append({

            "predicted_probability": model.get(
                "predicted_p_open",
                np.nan
            ),

            "actual_open": ground_truth.get(
                "actual_open",
                np.nan
            ),

            "actual_open_score": ground_truth.get(
                "future_open_score",
                np.nan
            ),

            "future_rain_score": ground_truth.get(
                "future_rain_score",
                np.nan
            ),

            "future_heat_score": ground_truth.get(
                "future_heat_score",
                np.nan
            ),
        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Remove invalid records
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "predicted_probability",
            "actual_open",
        ]
    )

    if len(df) == 0:
        raise ValueError(
            "No valid evaluation records found."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred_prob = df[
        "predicted_probability"
    ].to_numpy()

    y_true = df[
        "actual_open"
    ].astype(int).to_numpy()

    y_pred = (
        y_pred_prob >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # CLOSED accuracy
    # --------------------------------------------------------

    closed_mask = (
        y_true == 0
    )

    if closed_mask.sum() > 0:

        closed_accuracy = (
            y_pred[closed_mask] == 0
        ).mean()

    else:

        closed_accuracy = 0.0

    # --------------------------------------------------------
    # OPEN accuracy
    # --------------------------------------------------------

    open_mask = (
        y_true == 1
    )

    if open_mask.sum() > 0:

        open_accuracy = (
            y_pred[open_mask] == 1
        ).mean()

    else:

        open_accuracy = 0.0

    # --------------------------------------------------------
    # Additional event information
    # --------------------------------------------------------

    rain_minutes = (
        df["future_rain_score"] >= 0.5
    ).sum()

    heat_minutes = (
        df["future_heat_score"] >= 0.5
    ).sum()

    open_minutes = (
        y_true == 1
    ).sum()

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "fold": data.get(
            "fold",
            None
        ),

        "rain_minutes": int(
            rain_minutes
        ),

        "heat_minutes": int(
            heat_minutes
        ),

        "open_minutes": int(
            open_minutes
        ),

        "accuracy": accuracy,

        "closed_accuracy": closed_accuracy,

        "open_accuracy": open_accuracy,

        "open_recall": recall,

        "open_precision": precision,

        "f1": f1,
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
                f"Heat={result['heat_minutes']:4d} | "
                f"Open={result['open_minutes']:4d} | "
                f"Accuracy={result['accuracy']:.3f} | "
                f"Closed={result['closed_accuracy']:.3f} | "
                f"Recall={result['open_recall']:.3f} | "
                f"Precision={result['open_precision']:.3f} | "
                f"F1={result['f1']:.3f}"
            )

        except Exception as e:

            print(
                f"[ERROR] Fold {fold}: "
                f"{e}"
            )

    return pd.DataFrame(results)