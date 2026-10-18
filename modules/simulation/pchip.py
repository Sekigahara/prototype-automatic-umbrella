from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import PchipInterpolator

Mode = Literal["point", "mean", "total"]
Bounds = tuple[Optional[float], Optional[float]]

__all__ = ["upsample_weather", "upsample_points", "upsample_means", "upsample_totals"]

def _bounds_for(column: str, bounds: Mapping[str, Bounds]) -> tuple[float, float]:
    pair = bounds.get(column, (None, None))
    if len(pair) != 2:
        raise ValueError(f"{column}: bounds must contain [lower, upper].")
    lower = -np.inf if pair[0] is None else float(pair[0])
    upper = np.inf if pair[1] is None else float(pair[1])
    if np.isnan(lower) or np.isnan(upper) or lower > upper:
        raise ValueError(f"{column}: invalid bounds {pair}.")
    if lower == np.inf or upper == -np.inf:
        raise ValueError(f"{column}: bounds contain an invalid infinity.")
    return lower, upper

def _mean_blocks(
    y: np.ndarray, samples_per_minute: int, step_seconds: int,
    lower: float, upper: float,
) -> tuple[np.ndarray, int]:
    """Propose a shape, enforce each minute mean, then shrink only if needed.

    Treat minute means as values at minute centers ONLY to propose a shape.
    Evaluate that proposal at subinterval midpoints. Beyond the first/last
    center, hold the nearest center value, within the observed minute windows.
    This edge rule is an implementation assumption, not recovered information.
    """
    n = len(y)
    centers = np.arange(n, dtype=float) * 60.0 + 30.0
    midpoints = np.arange(n * samples_per_minute, dtype=float) * step_seconds
    midpoints += step_seconds / 2.0
    if n == 1:
        proposal = np.full((1, samples_per_minute), y[0])
    else:
        pchip = PchipInterpolator(centers, y, extrapolate=False)
        queries = np.clip(midpoints, centers[0], centers[-1])
        proposal = pchip(queries).reshape(n, samples_per_minute)

    # Equivalent to the report's additive mean-consistency correction.
    deviations = proposal - proposal.mean(axis=1, keepdims=True)

    # Extra engineering safeguard: scale zero-mean deviations, not the mean.
    # Unlike clipping the finished output, this retains minute-mean consistency.
    scale = np.ones(n, dtype=float)
    d_min = deviations.min(axis=1)
    d_max = deviations.max(axis=1)
    if np.isfinite(lower):
        mask = d_min < 0
        scale[mask] = np.minimum(scale[mask], (y[mask] - lower) / -d_min[mask])
    if np.isfinite(upper):
        mask = d_max > 0
        scale[mask] = np.minimum(scale[mask], (upper - y[mask]) / d_max[mask])
    scale = np.clip(scale, 0.0, 1.0)
    blocks = y[:, None] + scale[:, None] * deviations
    
    return blocks, int(np.count_nonzero(scale < 1.0))

def upsample_weather(
    data: pd.DataFrame,
    modes: Mapping[str, Mode],
    timestamp_column: str = "timestamp",
    step_seconds: int = 5,
    interval_label: Literal["start", "end"] = "start",
    bounds: Mapping[str, Bounds] | None = None,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a nonempty pandas DataFrame.")
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    if not isinstance(modes, Mapping) or not modes:
        raise ValueError("Provide a nonempty column-to-mode mapping.")
    if any(m not in ("point", "mean", "total") for m in modes.values()):
        raise ValueError("Each mode must be 'point', 'mean', or 'total'.")
    if (isinstance(step_seconds, bool) or not isinstance(step_seconds, Integral)
            or step_seconds <= 0 or 60 % step_seconds != 0):
        raise ValueError("step_seconds must be a positive integer divisor of 60.")
    step_seconds = int(step_seconds)
    if interval_label not in ("start", "end"):
        raise ValueError("interval_label must be 'start' or 'end'.")
    if timestamp_column in modes:
        raise ValueError("The timestamp column cannot also be a measurement.")
    missing = ({timestamp_column} | set(modes)) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    bounds = {} if bounds is None else bounds
    if set(bounds) - set(modes):
        raise ValueError("bounds contains a column not selected in modes.")
    if pd.api.types.is_numeric_dtype(data[timestamp_column]):
        raise ValueError("Decode numeric timestamps with an explicit unit first.")
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(data[timestamp_column], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid or mixed-timezone timestamps. Normalize first.") from exc
    if idx.hasnans or idx.has_duplicates:
        raise ValueError("Timestamps must be nonmissing and unique.")
    order = np.argsort(idx)
    idx = idx.take(order)
    frame = data.iloc[order]
    t = (idx - idx[0]).total_seconds().to_numpy(dtype=float)
    if len(idx) > 1 and not np.all(np.diff(t) == 60.0):
        raise ValueError("Expected contiguous 60-second records. Split gaps first.")
    if len(idx) < 2 and "point" in modes.values():
        raise ValueError("Point interpolation needs at least two observations.")
    
    count = 60 // step_seconds
    interval_start = idx[0] - pd.Timedelta(seconds=60 if interval_label == "end" else 0)
    point_grid = pd.date_range(idx[0], idx[-1], freq=pd.Timedelta(seconds=step_seconds))
    interval_grid = pd.date_range(interval_start, periods=len(idx) * count,
                                  freq=pd.Timedelta(seconds=step_seconds))
    outputs = []
    checks = {}
    adjustments = {}
    for column, mode in modes.items():
        y = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f"{column}: NaN or infinite values are not allowed.")
        if mode == "total" and column in bounds:
            raise ValueError(f"{column}: total mode does not accept bounds.")
        lower, upper = _bounds_for(column, bounds)
        if np.any(y < lower) or np.any(y > upper):
            raise ValueError(f"{column}: source values violate the supplied bounds.")
        adjustments[column] = 0
        if mode == "point":
            query = (point_grid - idx[0]).total_seconds().to_numpy(dtype=float)
            fine = PchipInterpolator(t, y, extrapolate=False)(query)
            fine[::count] = y  # retain original float values at observed timestamps
            grid = point_grid
            recovered = fine[::count]
        elif mode == "mean":
            blocks, adjustments[column] = _mean_blocks(y, count, step_seconds, lower, upper)
            fine, grid = blocks.ravel(), interval_grid
            recovered = blocks.mean(axis=1)
        else:
            if np.any(y < 0):
                raise ValueError(f"{column}: accumulated amounts cannot be negative.")
            blocks = np.repeat((y / count)[:, None], count, axis=1)
            fine, grid = blocks.ravel(), interval_grid
            recovered = blocks.sum(axis=1)
        if not np.isfinite(fine).all():
            raise ArithmeticError(f"{column}: nonfinite numerical output.")
        error = float(np.max(np.abs(recovered - y)))
        if not np.allclose(recovered, y, rtol=1e-12, atol=1e-10):
            raise ArithmeticError(f"{column}: failed consistency check ({error:g}).")
        tolerance = 1e-10 + 1e-12 * np.max(np.abs(y))
        if np.any(fine < lower - tolerance) or np.any(fine > upper + tolerance):
            raise ArithmeticError(f"{column}: generated values violate bounds.")
        checks[column] = error
        outputs.append(pd.Series(fine, index=grid, name=column))

    result = pd.concat(outputs, axis=1).sort_index()
    result.index.name = timestamp_column
    if result.isna().any().any():
        warnings.warn("Different point/interval coverage: uncovered edge values remain NaN. "
                      "Do not drop aggregate rows when checking minute conservation.",
                      UserWarning, stacklevel=2)
    result.attrs = {
        "product": "deterministically interpolated/disaggregated data; not measured 5s truth",
        "source_seconds": 60, "step_seconds": step_seconds,
        "input_interval_label": interval_label,
        "output_interval_label": "start", "modes": dict(modes),
        "bounds": {k: list(v) for k, v in bounds.items()},
        "input_rows": len(idx), "output_rows": len(result),
        "max_consistency_error": checks,
        "mean_bound_adjusted_minutes": adjustments,
        "mean_proposal": "PCHIP at source centers evaluated at bin midpoints; constant edge hold",
        "mean_limitation": "Mean correction/shrinkage can introduce minute-boundary jumps.",
        "total_assumption": "Uniform within-minute amount allocation; no recovered event timing.",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    return result

def _column_modes(columns: Sequence[str] | str, mode: Mode) -> dict[str, Mode]:
    """Validate selected columns and build a homogeneous mode mapping."""
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, Sequence):
        raise TypeError("columns must be a column name or a sequence of names.")
    if not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("columns must contain at least one nonempty column name.")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicate names.")
    return {column: mode for column in columns}


def upsample_points(
    data: pd.DataFrame,
    columns: Sequence[str] | str,
    timestamp_column: str = "timestamp",
    step_seconds: int = 5,
    bounds: Mapping[str, Bounds] | None = None,
) -> pd.DataFrame:
    """PCHIP-interpolate instantaneous one-minute readings.

    Returns 12*(N-1)+1 rows at the default five-second step. Stops at the last
    observation, preserves point values, and never extrapolates. Use this for
    point readings, NOT minute averages, rainfall totals, maxima, or angles.
    See upsample_weather for validation rules and return-index conventions.
    """
    return upsample_weather(
        data,
        modes=_column_modes(columns, "point"),
        timestamp_column=timestamp_column,
        step_seconds=step_seconds,
        bounds=bounds,
    )


def upsample_means(
    data: pd.DataFrame,
    columns: Sequence[str] | str,
    timestamp_column: str = "timestamp",
    step_seconds: int = 5,
    interval_label: Literal["start", "end"] = "start",
    bounds: Mapping[str, Bounds] | None = None,
) -> pd.DataFrame:
    """Disaggregate nonoverlapping minute averages, preserving each mean.

    Returns 12*N five-second interval averages by default. Proposes a PCHIP
    shape at interval midpoints (constant edge holds), corrects each minute
    mean, and optionally shrinks deviations to respect supplied bounds.
    These last two steps can introduce minute-boundary jumps. Output index
    labels are interval starts, even when the input is labeled by interval end.
    See upsample_weather for validation rules and observation semantics.
    """
    return upsample_weather(
        data,
        modes=_column_modes(columns, "mean"),
        timestamp_column=timestamp_column,
        step_seconds=step_seconds,
        interval_label=interval_label,
        bounds=bounds,
    )


def upsample_totals(
    data: pd.DataFrame,
    columns: Sequence[str] | str,
    timestamp_column: str = "timestamp",
    step_seconds: int = 5,
    interval_label: Literal["start", "end"] = "start",
) -> pd.DataFrame:
    """Equally divide nonnegative minute amounts into shorter intervals.

    Returns 12*N five-second amounts by default. Their within-minute sums
    reproduce the source amounts within floating-point tolerance. Assumes
    uniform allocation, not recovered event timing. For rainfall, input must
    be mm accumulated IN each minute, not mm/hour or a running cumulative
    counter. Output index labels are interval starts; see upsample_weather.
    """
    return upsample_weather(
        data,
        modes=_column_modes(columns, "total"),
        timestamp_column=timestamp_column,
        step_seconds=step_seconds,
        interval_label=interval_label,
    )
    