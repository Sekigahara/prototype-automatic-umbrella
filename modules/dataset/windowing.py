"""Trailing-window features at the native 20-second step.

Every row keeps its timestamp and gains columns that summarise the past few
minutes of sensor data, ending at that row. Windows look backwards only and
restart at every data gap (``segment``), so each value is exactly what the
device could compute from its own sensor buffer at that moment.

Compute these on the full continuous stream *before* splitting: evaluation
rows then get their real history (the minutes before an evaluation hour are
sensor readings the device would have), and no labels are involved. Keep the
longest window no longer than the split's gap so a training row's history
never reaches into an evaluation hour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["WINDOW_FEATURES", "add_window_features"]

# Sensors on the device: temperature, humidity, wind speed and direction, a
# rain detector (wet/dry) and a clock. No barometer and no rain gauge.
WINDOW_FEATURES = [
    "utci_mean_10m", "utci_max_10m",       # heat: the label uses a 10-min mean UTCI
    "wet_share_5m", "wet_share_10m",       # how much of the recent past was wet
    "rain_recency",                        # 1 while wet, fading like the label after rain
    "humidity", "humidity_chg_15m",        # rising humidity often precedes rain
    "wind_mean_5m", "wind_max_5m",         # steady wind vs gusts
    "hour_sin", "hour_cos",                # heat only counts in daytime
]

# Longest look-back used above; must not exceed the split's gap.
MAX_WINDOW = pd.Timedelta(minutes=15)
WET_CAP_MINUTES = 30.0
RAIN_DECAY_MINUTES = 5.0  # same fade as the rain part of the label


def _change(series: pd.Series, segment: pd.Series, steps: int) -> pd.Series:
    """Value now minus value ``steps`` rows earlier in the same segment.

    Early in a segment, where that row does not exist yet, the change is taken
    from the segment's first value instead.
    """
    earlier = series.groupby(segment).shift(steps)
    first = series.groupby(segment).transform("first")
    return series - earlier.fillna(first)


def _rolling(series: pd.Series, segment: pd.Series, window: str, how: str) -> pd.Series:
    rolled = getattr(series.groupby(segment).rolling(window, min_periods=1), how)()
    return rolled.reset_index(level=0, drop=True).reindex(series.index)


def add_window_features(df: pd.DataFrame, wet_columns: tuple[str, ...] = ("rain_flag",)) -> pd.DataFrame:
    """Add ``WINDOW_FEATURES`` (except the time ones) plus a ``warmup`` flag.

    ``df`` needs a regular DatetimeIndex, ``segment`` (from ``load_simulated``)
    and the columns utci, humidity, wind_speed and the columns in
    ``wet_columns`` (any value > 0 means wet; default: the rain detector only).
    ``warmup`` is True for the first ``MAX_WINDOW`` of each segment, where the
    longest windows are still filling up. ``min_since_wet`` (capped at 30) is
    kept for inspection; the model uses ``rain_recency`` instead, which spends
    its range on the first minutes after rain, where the label decides to close.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or "segment" not in df:
        raise ValueError("df needs a DatetimeIndex and a 'segment' column.")
    step = df.index.to_series().diff().median()
    segment = df["segment"]
    wet = (df[list(wet_columns)] > 0).any(axis=1).astype(float)

    times = df.index.to_series()
    last_wet = times.where(wet.astype(bool)).groupby(segment).ffill()
    minutes_since = ((times - last_wet).dt.total_seconds() / 60.0).clip(upper=WET_CAP_MINUTES)
    minutes_since = minutes_since.fillna(WET_CAP_MINUTES)
    segment_start = times.groupby(segment).transform("first")

    return df.assign(
        utci_mean_10m=_rolling(df["utci"], segment, "10min", "mean"),
        utci_max_10m=_rolling(df["utci"], segment, "10min", "max"),
        wet_share_5m=_rolling(wet, segment, "5min", "mean"),
        wet_share_10m=_rolling(wet, segment, "10min", "mean"),
        min_since_wet=minutes_since,
        rain_recency=np.exp(-minutes_since / RAIN_DECAY_MINUTES).where(minutes_since < WET_CAP_MINUTES, 0.0),
        humidity_chg_15m=_change(df["humidity"], segment, int(pd.Timedelta(minutes=15) / step)),
        wind_mean_5m=_rolling(df["wind_speed"], segment, "5min", "mean"),
        wind_max_5m=_rolling(df["wind_speed"], segment, "5min", "max"),
        warmup=(times - segment_start) < MAX_WINDOW,
    )
