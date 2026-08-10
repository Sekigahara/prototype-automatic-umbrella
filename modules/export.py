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
            "timestamp": str(row["Date/Time"]),

            # Original weather parameters
            "features": {
                "temperature_c": row["Temperature (°C)"],
                "precipitation_mm": row[
                    "1-minute Precipitation (mm)"
                ],
                "precipitation_presence": row[
                    "Precipitation Presence (Presence/Absence)"
                ],
                "wind_direction_deg": row[
                    "Wind Direction (deg)"
                ],
                "wind_speed_ms": row[
                    "Wind Speed (m/s)"
                ],
                "pressure_hpa": row[
                    "Local Pressure (hPa)"
                ],
                "humidity_percent": row[
                    "Humidity (%)"
                ],
            },

            # HMM engineered features
            "hmm_features": {
                "wind_dir_sin": row.get(
                    "wind_dir_sin",
                    None
                ),
                "wind_dir_cos": row.get(
                    "wind_dir_cos",
                    None
                ),
                "humidity_change_5min": row.get(
                    "humidity_change_5min",
                    None
                ),
                "pressure_change_5min": row.get(
                    "pressure_change_5min",
                    None
                ),
                "wind_change_5min": row.get(
                    "wind_change_5min",
                    None
                ),
                "rain_5min": row.get(
                    "rain_5min",
                    None
                ),
            },

            # Model output
            "hmm_state": int(
                row["HMM_State"]
            ),

            "predicted_probability": float(
                row["P_OPEN"]
            ),

            "predicted_open": int(
                row["P_OPEN"] >= 0.5
            ),

            # Ground truth
            "actual_future_rain_score": float(
                row["Actual_Soft_Label"]
            ),

            "actual_rain": int(
                row["Actual_Soft_Label"] >= 0.5
            ),
        }

        records.append(record)

    # 1. If 'metrics' is a 1-row DataFrame or Series, extract the row
    row = metrics.iloc[0] if isinstance(metrics, pd.DataFrame) else metrics

    # 2. Convert to dictionary while safely mapping types
    clean_metrics = pandas_to_metrics_dict(metrics)

    #print(metrics, type(metrics))
    output = {
        "fold": fold,

        "training": {
            "start": str(train_start),
            "end": str(train_end),
        },

        "evaluation": {
            "start": str(eval_start),
            "end": str(eval_end),
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