import numpy as np

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

def get_hmm_features():
    return [
        "Temperature (°C)",
        "1-minute Precipitation (mm)",
        "Precipitation Presence (Presence/Absence)",
        "wind_dir_sin",
        "wind_dir_cos",
        "Wind Speed (m/s)",
        "Local Pressure (hPa)",
        "Humidity (%)",
        "rain_5min",
        "humidity_change_5min",
        "pressure_change_5min",
        "wind_change_5min",
    ]

def train_hmm(
    train_df,
    previous_hmm=None,
    n_states=5,
    n_iter=50,
):

    features = get_hmm_features()

    # --------------------------------------------------------
    # Select valid rows
    # --------------------------------------------------------

    valid = (
        train_df[features]
        .notna()
        .all(axis=1)
    )

    train_valid = train_df.loc[
        valid
    ].copy()

    X = train_valid[
        features
    ].to_numpy(dtype=np.float64)

    # --------------------------------------------------------
    # Remove Inf
    # --------------------------------------------------------

    finite_mask = np.isfinite(X).all(axis=1)

    X = X[finite_mask]

    if len(X) == 0:
        raise ValueError(
            "No valid training samples."
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Final safety check
    if not np.isfinite(X_scaled).all():

        raise ValueError(
            "NaN/Inf detected after StandardScaler."
        )

    n_features = X_scaled.shape[1]

    # --------------------------------------------------------
    # Create HMM
    # --------------------------------------------------------

    if previous_hmm is None:

        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-3,
            random_state=42,
            min_covar=1e-3,
        )

    else:

        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-3,
            random_state=42,
            min_covar=1e-3,
            init_params="",
        )

        # --------------------------------------------
        # Previous start probabilities
        # --------------------------------------------

        startprob = (
            previous_hmm.startprob_
            .copy()
        )

        # Normalize safely
        startprob = np.nan_to_num(
            startprob,
            nan=1.0 / n_states,
            posinf=1.0 / n_states,
            neginf=1.0 / n_states,
        )

        startprob = np.maximum(
            startprob,
            1e-8
        )

        startprob /= startprob.sum()

        hmm.startprob_ = startprob

        # --------------------------------------------
        # Transition matrix
        # --------------------------------------------

        transmat = (
            previous_hmm.transmat_
            .copy()
        )

        transmat = np.nan_to_num(
            transmat,
            nan=1.0 / n_states,
            posinf=1.0 / n_states,
            neginf=1.0 / n_states,
        )

        transmat = np.maximum(
            transmat,
            1e-8
        )

        transmat /= transmat.sum(
            axis=1,
            keepdims=True
        )

        hmm.transmat_ = transmat

        # --------------------------------------------
        # Means
        # --------------------------------------------

        means = (
            previous_hmm.means_
            .copy()
        )

        if (
            means.shape
            != (n_states, n_features)
        ):
            raise ValueError(
                f"Invalid means shape: "
                f"{means.shape}"
            )

        means = np.nan_to_num(
            means,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        hmm.means_ = means

        # --------------------------------------------
        # Covariance
        # --------------------------------------------

        covars = (
            previous_hmm.covars_
            .copy()
        )

        # hmmlearn may expose diag covariance
        # as full matrices
        if covars.ndim == 3:

            covars = np.diagonal(
                covars,
                axis1=1,
                axis2=2
            )

        if covars.shape != (
            n_states,
            n_features
        ):
            raise ValueError(
                f"Invalid covariance shape: "
                f"{covars.shape}"
            )

        covars = np.nan_to_num(
            covars,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        # Prevent zero/negative variance
        covars = np.maximum(
            covars,
            1e-3
        )

        hmm.covars_ = covars

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    hmm.fit(X_scaled)

    # --------------------------------------------------------
    # Validate trained model
    # --------------------------------------------------------

    parameters = [
        hmm.startprob_,
        hmm.transmat_,
        hmm.means_,
        hmm.covars_,
    ]

    if not all(
        np.isfinite(x).all()
        for x in parameters
    ):
        raise ValueError(
            "HMM training produced NaN/Inf parameters."
        )

    # Check start probabilities
    if not np.isclose(
        hmm.startprob_.sum(),
        1.0
    ):
        hmm.startprob_ /= (
            hmm.startprob_.sum()
        )

    # Check transition probabilities
    row_sums = (
        hmm.transmat_.sum(axis=1)
    )

    hmm.transmat_ /= (
        row_sums[:, None]
    )

    return hmm, scaler

def learn_state_scores(
    train_df,
    hmm,
    scaler,
):
    features = get_hmm_features()

    valid = (
        train_df[features]
        .notna()
        .all(axis=1)
        &
        train_df["future_open_score"]
        .notna()
    )

    data = train_df.loc[
        valid
    ].copy()

    X = data[
        features
    ].to_numpy()

    X_scaled = scaler.transform(X)

    data["HMM_State"] = hmm.predict(
        X_scaled
    )

    state_scores = (
        data
        .groupby("HMM_State")
        ["future_open_score"]
        .mean()
    )

    default_score = (
        state_scores.mean()
        if len(state_scores)
        else 0.0
    )

    return np.array([
        state_scores.get(
            state,
            default_score
        )
        for state in range(
            hmm.n_components
        )
    ])

def evaluate_day(
    eval_df,
    hmm,
    scaler,
    state_scores,
):
    """
    Evaluate at full 1-minute resolution.
    """

    features = get_hmm_features()

    valid = (
        eval_df[features]
        .notna()
        .all(axis=1)
        &
        eval_df["future_open_score"]
        .notna()
    )

    result = eval_df.loc[
        valid
    ].copy()

    X = result[
        features
    ].to_numpy()

    X_scaled = scaler.transform(X)

    # ========================================================
    # STATE PROBABILITIES
    # ========================================================

    state_probs = hmm.predict_proba(
        X_scaled
    )

    result["HMM_State"] = hmm.predict(
        X_scaled
    )

    # ========================================================
    # STOCHASTIC P(OPEN)
    # ========================================================

    result["P_OPEN"] = (
        state_probs @ state_scores
    ).clip(0, 1)

    result["P_CLOSE"] = (
        1 - result["P_OPEN"]
    )

    # ========================================================
    # ACTUAL LABEL
    # ========================================================

    result["Actual_Soft_Label"] = (
        result["future_open_score"]
    )

    # Keep individual components for analysis
    result["Actual_Rain_Score"] = (
        result["future_rain_score"]
    )

    result["Actual_Heat_Score"] = (
        result["future_heat_score"]
    )

    # ========================================================
    # HARD ACTION
    # ========================================================

    result["Predicted_Action"] = np.where(
        result["P_OPEN"] >= 0.5,
        "OPEN",
        "CLOSE",
    )

    # ========================================================
    # STORE STATE PROBABILITIES
    # ========================================================

    for state in range(
        hmm.n_components
    ):
        result[
            f"State_{state}_Prob"
        ] = state_probs[:, state]

    return result