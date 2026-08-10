import numpy as np
import pandas as pd

def prepare_data(weather_data):
    WEATHER_COLS = [
        "Temperature (°C)",
        "1-minute Precipitation (mm)",
        "Precipitation Presence (Presence/Absence)",
        "Wind Direction (deg)",
        "Wind Speed (m/s)",
        "Local Pressure (hPa)",
        "Humidity (%)",
    ]

    df = pd.concat(
        weather_data.values(),
        ignore_index=True
    ).copy()

    df["Date/Time"] = pd.to_datetime(
        df["Date/Time"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["Date/Time"])
        .sort_values("Date/Time")
        .reset_index(drop=True)
    )

    # Clean numeric weather columns
    for col in WEATHER_COLS:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({
                "": np.nan,
                "-": np.nan,
                "--": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
            })
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df

def create_features(df, HORIZON=5):
    df = df.copy()

    # Wind direction
    wind_rad = np.deg2rad(
        pd.to_numeric(
            df["Wind Direction (deg)"],
            errors="coerce"
        )
    )

    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    # 5-minute historical changes
    df["humidity_change_5min"] = (
        df["Humidity (%)"]
        - df["Humidity (%)"].shift(HORIZON)
    )

    df["pressure_change_5min"] = (
        df["Local Pressure (hPa)"]
        - df["Local Pressure (hPa)"].shift(HORIZON)
    )

    df["wind_change_5min"] = (
        df["Wind Speed (m/s)"]
        - df["Wind Speed (m/s)"].shift(HORIZON)
    )

    # Past 5-minute rain
    precipitation = (
        pd.to_numeric(
            df["Precipitation Presence (Presence/Absence)"],
            errors="coerce"
        )
        .eq(10)
        .astype(float)
    )

    df["rain_5min"] = (
        precipitation
        .shift(1)
        .rolling(HORIZON)
        .mean()
    )

    # Replace NaN / inf in engineered features
    engineered_cols = [
        "wind_dir_sin",
        "wind_dir_cos",
        "humidity_change_5min",
        "pressure_change_5min",
        "wind_change_5min",
        "rain_5min",
    ]

    df[engineered_cols] = (
        df[engineered_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df

def create_future_label(df, horizon=5):
    df = df.copy()

    # 10 = precipitation present
    rain = (
        df["Precipitation Presence (Presence/Absence)"]
        .fillna(0)
        .eq(10)
        .astype(float)
    )

    # Future precipitation probability over next HORIZON minutes
    future_rain = sum(
        rain.shift(-i)
        for i in range(1, horizon + 1)
    )

    df["future_rain_score"] = (
        future_rain / horizon
    )

    # Remove rows where the complete future horizon
    # does not exist
    df.loc[
        df["future_rain_score"].isna(),
        "future_rain_score"
    ] = np.nan

    return df