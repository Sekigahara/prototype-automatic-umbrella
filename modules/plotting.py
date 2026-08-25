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
    # OVERALL OPEN / CLOSE PERFORMANCE
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

    axes[0].plot(
        x,
        results["open_accuracy"],
        marker="o",
        label="Open Accuracy"
    )

    axes[0].set_ylabel(
        "Score"
    )

    axes[0].set_ylim(
        0,
        1.05
    )

    axes[0].set_title(
        "Daily Umbrella OPEN / CLOSE Performance"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # ========================================================
    # OPEN DETECTION
    # ========================================================

    axes[1].plot(
        x,
        results["open_recall"],
        marker="o",
        label="Open Recall"
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
        "Umbrella OPEN Detection Performance"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    # ========================================================
    # EVENT DISTRIBUTION
    # ========================================================

    axes[2].plot(
        x,
        results["rain_minutes"],
        marker="o",
        label="Rain-related minutes"
    )

    axes[2].plot(
        x,
        results["heat_minutes"],
        marker="o",
        label="Heat-related minutes"
    )

    axes[2].plot(
        x,
        results["open_minutes"],
        marker="o",
        label="Total OPEN minutes"
    )

    axes[2].set_xlabel(
        "Fold / Evaluation Day"
    )

    axes[2].set_ylabel(
        "Minutes"
    )

    axes[2].set_title(
        "OPEN Condition Distribution"
    )

    axes[2].legend()

    axes[2].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

def plot_open_distribution(results):

    plt.figure(
        figsize=(16, 5)
    )

    x = results["fold"]

    plt.plot(
        x,
        results["rain_minutes"],
        marker="o",
        label="Rain minutes"
    )

    plt.plot(
        x,
        results["heat_minutes"],
        marker="o",
        label="Heat minutes"
    )

    plt.plot(
        x,
        results["open_minutes"],
        marker="o",
        label="OPEN minutes"
    )

    plt.xlabel(
        "Fold / Evaluation Day"
    )

    plt.ylabel(
        "Minutes"
    )

    plt.title(
        "Rain, Heat, and OPEN Conditions per Evaluation Day"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

def plot_open_folds(
    results,
    log_dir,
    threshold=0.5,
    max_folds=None
):

    # ========================================================
    # SELECT FOLDS THAT CONTAIN OPEN CONDITIONS
    # ========================================================

    open_folds = results[
        results["open_minutes"] > 0
    ].copy()

    if max_folds is not None:

        open_folds = open_folds.head(
            max_folds
        )

    # ========================================================
    # LOOP THROUGH FOLDS
    # ========================================================

    for _, row in open_folds.iterrows():

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

        records = []

        # ====================================================
        # FLATTEN NESTED JSON
        # ====================================================

        for record in data["records"]:

            model = record.get(
                "model",
                {}
            )

            ground_truth = record.get(
                "ground_truth",
                {}
            )

            heat_info = record.get(
                "heat_information",
                {}
            )

            records.append({

                "timestamp": record.get(
                    "timestamp"
                ),

                # Prediction
                "predicted_probability": model.get(
                    "predicted_p_open"
                ),

                "predicted_action": model.get(
                    "predicted_action"
                ),

                # Ground truth
                "actual_open": ground_truth.get(
                    "actual_open"
                ),

                "future_open_score": ground_truth.get(
                    "future_open_score"
                ),

                "future_rain_score": ground_truth.get(
                    "future_rain_score"
                ),

                "future_heat_score": ground_truth.get(
                    "future_heat_score"
                ),

                # Heat debug
                "effective_heat": heat_info.get(
                    "effective_heat"
                ),

                "heat_score": heat_info.get(
                    "heat_score"
                ),
            })

        df = pd.DataFrame(
            records
        )

        # ====================================================
        # TYPE CONVERSION
        # ====================================================

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        numeric_cols = [
            "predicted_probability",
            "actual_open",
            "future_open_score",
            "future_rain_score",
            "future_heat_score",
            "effective_heat",
            "heat_score",
        ]

        for col in numeric_cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.sort_values(
            "timestamp"
        )

        # ====================================================
        # PREDICTED BINARY ACTION
        # ====================================================

        df["predicted_open"] = (
            df["predicted_probability"]
            >= threshold
        ).astype(int)

        # ====================================================
        # CLASSIFICATION ERROR
        # ====================================================

        wrong = (
            df["actual_open"]
            != df["predicted_open"]
        )

        error = wrong.astype(int)

        # ====================================================
        # PLOT
        # ====================================================

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 11),
            sharex=True,
            gridspec_kw={
                "height_ratios": [
                    2,
                    1.5,
                    1.5,
                    0.6
                ]
            }
        )

        # ====================================================
        # 1. P(OPEN) VS ACTUAL SOFT LABEL
        # ====================================================

        axes[0].plot(
            df["timestamp"],
            df["predicted_probability"],
            label="HMM P(OPEN)",
            linewidth=1
        )

        axes[0].plot(
            df["timestamp"],
            df["future_open_score"],
            label="Actual OPEN Score",
            linewidth=1
        )

        axes[0].axhline(
            threshold,
            linestyle="--",
            label="Decision Threshold"
        )

        axes[0].set_ylim(
            0,
            1
        )

        axes[0].set_ylabel(
            "Probability"
        )

        axes[0].set_title(
            f"Fold {fold:02d} | "
            f"Rain={row['rain_minutes']} min | "
            f"Heat={row['heat_minutes']} min | "
            f"OPEN={row['open_minutes']} min | "
            f"F1={row['f1']:.3f}"
        )

        axes[0].legend()

        axes[0].grid(
            alpha=0.3
        )

        # ====================================================
        # 2. RAIN + HEAT CONTRIBUTIONS
        # ====================================================

        axes[1].plot(
            df["timestamp"],
            df["future_rain_score"],
            label="Future Rain Score",
            linewidth=1
        )

        axes[1].plot(
            df["timestamp"],
            df["future_heat_score"],
            label="Future Heat Score",
            linewidth=1
        )

        axes[1].plot(
            df["timestamp"],
            df["future_open_score"],
            linestyle="--",
            label="Combined OPEN Score",
            linewidth=1.5
        )

        axes[1].set_ylim(
            0,
            1
        )

        axes[1].set_ylabel(
            "Soft Label"
        )

        axes[1].set_title(
            "Rain and Heat Contributions"
        )

        axes[1].legend()

        axes[1].grid(
            alpha=0.3
        )

        # ====================================================
        # 3. ACTUAL VS PREDICTED ACTION
        # ====================================================

        axes[2].step(
            df["timestamp"],
            df["actual_open"],
            where="post",
            label="Actual OPEN",
            linewidth=1.5
        )

        axes[2].step(
            df["timestamp"],
            df["predicted_open"],
            where="post",
            label="Predicted OPEN",
            linewidth=1.2
        )

        axes[2].set_yticks(
            [0, 1]
        )

        axes[2].set_yticklabels([
            "CLOSED",
            "OPEN"
        ])

        axes[2].set_ylabel(
            "Umbrella"
        )

        axes[2].set_title(
            "Actual vs Predicted Action"
        )

        axes[2].legend()

        axes[2].grid(
            alpha=0.3
        )

        # ====================================================
        # 4. ERROR
        # ====================================================

        axes[3].step(
            df["timestamp"],
            error,
            where="post",
            linewidth=1.5
        )

        axes[3].set_yticks(
            [0, 1]
        )

        axes[3].set_yticklabels([
            "Correct",
            "Wrong"
        ])

        axes[3].set_ylabel(
            "Error"
        )

        axes[3].set_xlabel(
            "Time"
        )

        axes[3].grid(
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
    "fold": data.get("fold"),

    "rain_minutes": int(rain_minutes),
    "heat_minutes": int(heat_minutes),
    "open_minutes": int(open_minutes),

    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,

    "accuracy": accuracy,
    "closed_accuracy": closed_accuracy,
    "open_accuracy": open_accuracy,
    "open_recall": recall,
    "open_precision": precision,
    "f1": f1,
}