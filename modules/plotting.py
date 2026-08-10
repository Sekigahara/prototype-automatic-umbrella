import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_fold_metrics(results):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True
    )

    x = results["fold"]

    # ========================================================
    # Overall / closed
    # ========================================================

    axes[0].plot(
        x,
        results["accuracy"],
        marker="o",
        label="Accuracy"
    )

    axes[0].plot(
        x,
        results["closed_accuracy"],
        marker="o",
        label="Closed Accuracy"
    )

    axes[0].set_ylabel(
        "Score"
    )

    axes[0].set_ylim(
        0,
        1.05
    )

    axes[0].set_title(
        "Daily Umbrella Decision Performance"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # ========================================================
    # Rain detection
    # ========================================================

    axes[1].plot(
        x,
        results["rain_recall"],
        marker="o",
        label="Rain Recall"
    )

    axes[1].plot(
        x,
        results["open_precision"],
        marker="o",
        label="Open Precision"
    )

    axes[1].plot(
        x,
        results["f1"],
        marker="o",
        label="F1"
    )

    axes[1].set_ylabel(
        "Score"
    )

    axes[1].set_ylim(
        0,
        1.05
    )

    axes[1].set_title(
        "Rain / Opening Detection Performance"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    # ========================================================
    # Probability quality
    # ========================================================

    axes[2].plot(
        x,
        results["roc_auc"],
        marker="o",
        label="ROC-AUC"
    )

    axes[2].plot(
        x,
        results["pr_auc"],
        marker="o",
        label="PR-AUC"
    )

    axes[2].plot(
        x,
        results["brier"],
        marker="o",
        label="Brier Score"
    )

    axes[2].set_xlabel(
        "Fold / Evaluation Day"
    )

    axes[2].set_ylabel(
        "Score"
    )

    axes[2].set_title(
        "Probability Quality"
    )

    axes[2].legend()

    axes[2].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

def plot_rain_distribution(results):

    plt.figure(
        figsize=(16, 5)
    )

    plt.bar(
        results["fold"],
        results["rain_minutes"]
    )

    plt.xlabel(
        "Fold / Evaluation Day"
    )

    plt.ylabel(
        "Rain minutes"
    )

    plt.title(
        "Rain Occurrence per Evaluation Day"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

def plot_rainy_folds(
    results,
    log_dir,
    threshold=0.5,
    max_folds=None
):

    rainy = results[
        results["rain_minutes"] > 0
    ].copy()

    if max_folds is not None:

        rainy = rainy.head(
            max_folds
        )

    for _, row in rainy.iterrows():

        fold = int(
            row["fold"]
        )

        path = (
            log_dir
            / f"fold_{fold:03d}.json"
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        df = pd.DataFrame(
            data["records"]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df["predicted_probability"] = pd.to_numeric(
            df["predicted_probability"],
            errors="coerce"
        )

        df["actual_rain"] = pd.to_numeric(
            df["actual_rain"],
            errors="coerce"
        )

        df = df.sort_values(
            "timestamp"
        )

        df["predicted_open"] = (
            df["predicted_probability"]
            >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(16, 8),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2, 2, 0.6]
            }
        )

        # Probability
        axes[0].plot(
            df["timestamp"],
            df["predicted_probability"],
            label="HMM P(Open)",
            linewidth=1
        )

        axes[0].axhline(
            threshold,
            linestyle="--",
            label="Threshold"
        )

        axes[0].fill_between(
            df["timestamp"],
            0,
            1,
            where=df["actual_rain"].astype(bool),
            alpha=0.2,
            label="Actual rain"
        )

        axes[0].set_ylim(
            0,
            1
        )

        axes[0].set_ylabel(
            "P(Open)"
        )

        axes[0].set_title(
            f"Fold {fold:02d} | "
            f"{row['date']} | "
            f"Rain={row['rain_minutes']} min | "
            f"F1={row['f1']:.3f}"
        )

        axes[0].legend()

        axes[0].grid(
            alpha=0.3
        )

        # ----------------------------------------------------
        # Actual vs prediction
        # ----------------------------------------------------

        axes[1].step(
            df["timestamp"],
            df["actual_rain"],
            where="post",
            label="Actual",
            linewidth=1.5
        )

        axes[1].step(
            df["timestamp"],
            df["predicted_open"],
            where="post",
            label="Predicted",
            linewidth=1.2
        )

        axes[1].set_yticks([0, 1])

        axes[1].set_yticklabels([
            "CLOSED",
            "OPEN"
        ])

        axes[1].set_ylabel(
            "Umbrella"
        )

        axes[1].legend()

        axes[1].grid(
            alpha=0.3
        )

        # Plot missclassificaiton
        wrong = (
            df["actual_rain"]
            != df["predicted_open"]
        )

        error = wrong.astype(int)

        axes[2].step(
            df["timestamp"],
            error,
            where="post",
            linewidth=2
        )

        axes[2].set_yticks(
            [0, 1]
        )

        axes[2].set_yticklabels([
            "Correct",
            "Wrong"
        ])

        axes[2].set_ylabel(
            "Error"
        )

        axes[2].set_xlabel(
            "Time"
        )

        axes[2].grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

def pooled_metrics(results):

    TP = results["TP"].sum()
    TN = results["TN"].sum()
    FP = results["FP"].sum()
    FN = results["FN"].sum()

    total = (
        TP +
        TN +
        FP +
        FN
    )

    accuracy = (
        (TP + TN) / total
    )

    closed_accuracy = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else np.nan
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else np.nan
    )

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else np.nan
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else np.nan
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,

        "Accuracy": accuracy,
        "Closed Accuracy": closed_accuracy,
        "Rain Recall": recall,
        "Open Precision": precision,
        "F1": f1,
    }