import json
import numpy as np
import pandas as pd

from pathlib import Path

def sanitize_value(v):
    """Recursively converts Pandas/NumPy objects into native JSON-serializable Python types."""
    if pd.isna(v):
        return None
    elif isinstance(v, (np.integer, int)):
        return int(v)
    elif isinstance(v, (np.floating, float)):
        return float(v)
    elif isinstance(v, (np.bool_, bool)):
        return bool(v)
    elif isinstance(v, (pd.Timestamp, np.datetime64)):
        return v.isoformat()
    elif isinstance(v, np.ndarray):
        return [sanitize_value(x) for x in v.tolist()]
    elif isinstance(v, dict):
        return {str(k): sanitize_value(val) for k, val in v.items()}
    return str(v) if not isinstance(v, (str, list)) else v


def pandas_to_metrics_dict(data):
    """Converts a Pandas DataFrame or Series into a JSON-serializable dictionary.

    - Handles single-row DataFrames -> returns a single dict - Handles
    multi-row DataFrames -> returns a list of dicts - Converts np.int64,
    np.float64, NaNs, and Timestamps to native Python types
    """
    # If it's a Series, convert directly
    if isinstance(data, pd.Series):
        return {k: sanitize_value(v) for k, v in data.items()}

    # If it's a DataFrame
    if isinstance(data, pd.DataFrame):
        # Case 1: Multiple rows -> returns list of dicts
        if len(data) > 1:
            return [
                {k: sanitize_value(v) for k, v in row.items()}
                for _, row in data.iterrows()
            ]

        # Case 2: Single row -> returns a single dict
        if len(data) == 1:
            return {k: sanitize_value(v) for k, v in data.iloc[0].items()}

        return {}

    # If it's already a dictionary
    if isinstance(data, dict):
        return {
            k: (
                sanitize_value(v.item())
                if hasattr(v, "item") and not isinstance(v, (str, list, dict))
                else sanitize_value(v)
            )
            for k, v in data.items()
        }

    return sanitize_value(data)

def save_evaluation_json(
    result,
    fold,
    train_start,
    train_end,
    eval_start,
    eval_end,
    metrics,
    output_dir="evaluation_logs",
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    for _, row in result.iterrows():

        record = {

            "timestamp": sanitize_value(
                row["Date/Time"]
            ),

            # ====================================================
            # ORIGINAL WEATHER PARAMETERS
            # ====================================================

            "features": {

                "temperature_c": sanitize_value(
                    row["Temperature (°C)"]
                ),

                "precipitation_mm": sanitize_value(
                    row["1-minute Precipitation (mm)"]
                ),

                "precipitation_presence": sanitize_value(
                    row[
                        "Precipitation Presence (Presence/Absence)"
                    ]
                ),

                "wind_direction_deg": sanitize_value(
                    row["Wind Direction (deg)"]
                ),

                "wind_speed_ms": sanitize_value(
                    row["Wind Speed (m/s)"]
                ),

                "pressure_hpa": sanitize_value(
                    row["Local Pressure (hPa)"]
                ),

                "humidity_percent": sanitize_value(
                    row["Humidity (%)"]
                ),
            },

            # ====================================================
            # HMM ENGINEERED FEATURES
            # ====================================================

            "hmm_features": {

                "wind_dir_sin": sanitize_value(
                    row.get("wind_dir_sin", None)
                ),

                "wind_dir_cos": sanitize_value(
                    row.get("wind_dir_cos", None)
                ),

                "humidity_change_5min": sanitize_value(
                    row.get(
                        "humidity_change_5min",
                        None
                    )
                ),

                "pressure_change_5min": sanitize_value(
                    row.get(
                        "pressure_change_5min",
                        None
                    )
                ),

                "wind_change_5min": sanitize_value(
                    row.get(
                        "wind_change_5min",
                        None
                    )
                ),

                "rain_5min": sanitize_value(
                    row.get(
                        "rain_5min",
                        None
                    )
                ),
            },

            # ====================================================
            # HMM MODEL OUTPUT
            # ====================================================

            "model": {

                "hmm_state": sanitize_value(
                    row["HMM_State"]
                ),

                "predicted_p_open": sanitize_value(
                    row["P_OPEN"]
                ),

                "predicted_p_close": sanitize_value(
                    row.get(
                        "P_CLOSE",
                        1 - row["P_OPEN"]
                    )
                ),

                "predicted_action": row.get(
                    "Predicted_Action",
                    "OPEN"
                    if row["P_OPEN"] >= 0.5
                    else "CLOSE"
                ),
            },

            # ====================================================
            # GROUND TRUTH COMPONENTS
            # ====================================================

            "ground_truth": {

                # Rain contribution
                "future_rain_score": sanitize_value(
                    row.get(
                        "future_rain_score",
                        row.get(
                            "Actual_Rain_Score",
                            None
                        )
                    )
                ),

                # Heat contribution
                "future_heat_score": sanitize_value(
                    row.get(
                        "future_heat_score",
                        row.get(
                            "Actual_Heat_Score",
                            None
                        )
                    )
                ),

                # Combined rain + heat OPEN target
                "future_open_score": sanitize_value(
                    row["Actual_Soft_Label"]
                ),

                # Binary OPEN/CLOSE reference
                "actual_action": (
                    "OPEN"
                    if row["Actual_Soft_Label"] >= 0.5
                    else "CLOSE"
                ),

                "actual_open": int(
                    row["Actual_Soft_Label"] >= 0.5
                ),
            },

            # ====================================================
            # OPTIONAL HEAT DEBUG INFORMATION
            # ====================================================

            "heat_information": {

                "effective_heat": sanitize_value(
                    row.get(
                        "effective_heat",
                        None
                    )
                ),

                "heat_score": sanitize_value(
                    row.get(
                        "heat_score",
                        None
                    )
                ),
            },
        }

        records.append(record)

    # ============================================================
    # CLEAN METRICS
    # ============================================================

    clean_metrics = pandas_to_metrics_dict(
        metrics
    )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    output = {

        "fold": sanitize_value(fold),

        "training": {
            "start": sanitize_value(train_start),
            "end": sanitize_value(train_end),
        },

        "evaluation": {
            "start": sanitize_value(eval_start),
            "end": sanitize_value(eval_end),
        },

        "metrics": clean_metrics,

        "records": records,
    }

    filename = (
        output_dir
        / f"fold_{fold:03d}.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return filename