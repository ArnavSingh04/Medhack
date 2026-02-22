"""
Feature extraction utilities for vital sign time-series and patient demographics.

Includes:
- Vectorized spectral feature computation (FFT-based)
- Patient feature builder (EDA-informed transforms, OHE, risk tiers)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_rolling_window(
    df: pd.DataFrame,
    col: str,
    n_window: int,
    encounter_id_col: str = "encounter_id",
) -> np.ndarray:
    """
    Build rolling window matrix (n_rows, n_window) from a vital column.
    Window = [t-(n_window-1), ..., t-1, t] (oldest first).
    Does not require lag columns; uses shift() on the fly.
    """
    # shifts: n_window-1 down to 0 (oldest to newest)
    shifts = list(range(n_window - 1, -1, -1))
    stacked = np.column_stack(
        [df.groupby(encounter_id_col)[col].shift(s).values for s in shifts]
    )
    return stacked.astype(np.float64)


def _impute_window_nans(
    window: np.ndarray,
    df: pd.DataFrame,
    col: str,
    encounter_id_col: str = "encounter_id",
) -> np.ndarray:
    """Fill NaN in window with encounter-level median of the vital."""
    encounter_median = df.groupby(encounter_id_col)[col].transform("median").values
    window = np.where(np.isnan(window), encounter_median[:, np.newaxis], window)
    return window


def _compute_spectral_features_vectorized(
    window: np.ndarray,
    fs: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dom_freq, power_low, autocorr for all rows in a window matrix.
    window: (n_rows, n_window), float64
    Returns: (dom_freq, power_low, autocorr) each shape (n_rows,)
    """
    n = window.shape[1]
    if n < 4:
        nan_arr = np.full(window.shape[0], np.nan, dtype=np.float64)
        return nan_arr, nan_arr, nan_arr

    # Impute any remaining NaNs with row median
    row_median = np.nanmedian(window, axis=1, keepdims=True)
    x = np.where(np.isnan(window), row_median, window)

    # Detrend and apply Hanning
    x_mean = x.mean(axis=1, keepdims=True)
    x_detrend = x - x_mean
    hanning = np.hanning(n)
    x_windowed = x_detrend * hanning

    # FFT and power
    fft_vals = np.fft.rfft(x_windowed, axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(fft_vals) ** 2
    power[:, 0] = 0  # DC not informative

    total_power = power.sum(axis=1)
    valid_power = total_power >= 1e-10

    # Dominant frequency (excluding DC)
    power_ac = power[:, 1:]
    dom_idx = np.argmax(power_ac, axis=1)
    dom_freq = freqs[1:][dom_idx].astype(np.float64)
    dom_freq = np.where(valid_power, dom_freq, np.nan)

    # Power in low band (0, 0.05] Hz
    low_mask = (freqs > 0) & (freqs <= 0.05)
    power_low = np.sum(power[:, low_mask], axis=1) / np.where(
        valid_power, total_power, np.nan
    )
    power_low = np.where(valid_power, power_low, np.nan)

    # Lag-1 autocorrelation (vectorized)
    x_centered = x - x_mean
    x0 = x_centered[:, :-1]
    x1 = x_centered[:, 1:]
    cov = np.sum(x0 * x1, axis=1)
    var0 = np.sum(x0**2, axis=1)
    var1 = np.sum(x1**2, axis=1)
    var_product = var0 * var1
    valid_autocorr = var_product >= 1e-20
    autocorr = np.where(
        valid_autocorr,
        cov / np.sqrt(var_product),
        np.nan,
    )

    return dom_freq, power_low, autocorr


def add_spectral_features(
    df: pd.DataFrame,
    vital_cols: list[str],
    n_window: int,
    fs: float = 0.2,
    encounter_id_col: str = "encounter_id",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add spectral features (dom_freq, power_low, autocorr) per vital using a
    vectorized FFT-based pipeline.

    No lag columns are required; the rolling window is built on-the-fly via
    groupby + shift, then FFT is applied in batch.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain encounter_id_col and vital_cols. Vitals should be imputed.
    vital_cols : list[str]
        Column names for vitals to process.
    n_window : int
        Number of points in the rolling window (e.g. 37 for 180s at 5s cadence).
    fs : float
        Sample rate in Hz (e.g. 0.2 for 5s interval).
    encounter_id_col : str
        Column used to group rows per encounter.

    Returns
    -------
    out : pd.DataFrame
        Copy of df with new columns appended.
    new_cols : list[str]
        Names of added columns.

    Examples
    --------
    >>> df, cols = add_spectral_features(df, ['heart_rate', 'oxygen_saturation'], 37, 0.2)
    >>> # Adds heart_rate_dom_freq_w37, heart_rate_power_low_w37, heart_rate_autocorr_w37, etc.
    """
    out = df.copy()
    new_cols: list[str] = []
    suffix = f"_w{n_window}"

    for col in vital_cols:
        if col not in out.columns:
            raise ValueError(f"Column '{col}' not in dataframe")

        window = _build_rolling_window(out, col, n_window, encounter_id_col)
        window = _impute_window_nans(window, out, col, encounter_id_col)

        dom_freq, power_low, autocorr = _compute_spectral_features_vectorized(
            window, fs
        )

        out[f"{col}_dom_freq{suffix}"] = dom_freq
        out[f"{col}_power_low{suffix}"] = power_low
        out[f"{col}_autocorr{suffix}"] = autocorr
        new_cols.extend([
            f"{col}_dom_freq{suffix}",
            f"{col}_power_low{suffix}",
            f"{col}_autocorr{suffix}",
        ])

    return out, new_cols


# ---------------------------------------------------------------------------
# Lag features (for derivatives, rolling stats, derived vitals)
# ---------------------------------------------------------------------------
# Spectral features build their 37-point window on-the-fly and do NOT use
# these lag columns. Lag1..lag12 are required by derivative features, rolling
# stats, and derived vitals in the pipeline.


def add_lag_features(
    df: pd.DataFrame,
    vital_cols: list[str],
    n_lags: int,
    encounter_id_col: str = "encounter_id",
) -> pd.DataFrame:
    """
    Add lag1..lag{n_lags} for each vital within each encounter.
    Leading rows (warmup) will have NaN — use add_lag_features_with_imputation
    to fill with encounter median.
    """
    out = df.sort_values([encounter_id_col, "timestamp"]).reset_index(
        drop=True
    ).copy()
    for col in vital_cols:
        g = out.groupby(encounter_id_col)[col]
        for lag in range(1, n_lags + 1):
            out[f"{col}_lag{lag}"] = g.shift(lag)
    return out


def impute_lag_features(
    df: pd.DataFrame,
    vital_cols: list[str],
    n_lags: int,
    encounter_id_col: str = "encounter_id",
) -> pd.DataFrame:
    """
    Fill NaN in lag columns (warmup rows) with encounter-level median of
    the vital. Reasonable when we must predict for warmup rows (e.g. holdout)
    and cannot drop them.
    """
    out = df.copy()
    for col in vital_cols:
        enc_median = out.groupby(encounter_id_col)[col].transform("median")
        for lag in range(1, n_lags + 1):
            lag_col = f"{col}_lag{lag}"
            out[lag_col] = out[lag_col].fillna(enc_median)
    return out


def add_lag_features_with_imputation(
    df: pd.DataFrame,
    vital_cols: list[str],
    n_lags: int,
    encounter_id_col: str = "encounter_id",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add lag1..lag{n_lags} and impute warmup NaNs with encounter median.
    Returns (df with lag cols, list of lag column names).
    """
    out = add_lag_features(df, vital_cols, n_lags, encounter_id_col)
    out = impute_lag_features(out, vital_cols, n_lags, encounter_id_col)
    lag_cols = [
        f"{col}_lag{lag}"
        for col in vital_cols
        for lag in range(1, n_lags + 1)
    ]
    return out, lag_cols


# ---------------------------------------------------------------------------
# Patient feature builder (EDA-informed, leakage-safe)
# ---------------------------------------------------------------------------

DROP_PATIENT_COLS = [
    "patient_name",
    "encounter_class",
    "date_of_birth",
    "known_allergies",
    "previous_medications",
]

NUMERIC_PAT_COLS = ["age", "bmi", "pain_score"]

HIGH_RISK_REASONS = [
    "Myocardial infarction",
    "Cerebrovascular accident",
    "Gunshot wound",
    "Pneumonia",
    "Sepsis",
    "Seizure disorder",
    "Drug overdose",
]

MEDIUM_RISK_REASONS = [
    "Appendicitis",
    "Chronic congestive heart failure",
    "Acute allergic reaction",
    "Childhood asthma",
    "Asthma",
    "Concussion",
    "Injury of neck",
    "Suspected lung cancer",
    "Acute bronchitis",
]

COMORBIDITY_KEYWORDS = {
    "has_hypertension": "hypertension",
    "has_diabetes": "diabetes|mellitus",
    "has_kidney": "kidney",
    "has_cardiac": "cardiac|heart|coronary|infarct",
    "has_anemia": "anemia",
    "has_obesity": "obesity",
    "has_chronic": "chronic",
}

CARDIAC_MED_KEYWORDS = (
    "metoprolol|nitroglycerin|clopidogrel|simvastatin|"
    "lisinopril|hydrochlorothiazide"
)


def _assign_reason_risk_tier(reason: str | float) -> int:
    """Map reason_for_visit to risk tier: 0=low, 1=medium, 2=high."""
    if pd.isna(reason):
        return 1  # missing → medium (conservative)
    reason_lower = str(reason).lower()
    for hr in HIGH_RISK_REASONS:
        if hr.lower() in reason_lower:
            return 2
    for mr in MEDIUM_RISK_REASONS:
        if mr.lower() in reason_lower:
            return 1
    return 0


class PatientFeatureBuilder:
    """
    Build patient-level features from patients.csv with leakage-safe fitting.
    Fits scalers/encoders on train encounters only, applies to full patients.
    """

    def __init__(self) -> None:
        self._num_scaler: StandardScaler | None = None
        self._gender_encoder: OneHotEncoder | None = None
        self._ms_encoder: OneHotEncoder | None = None
        self._cat_encoder: OneHotEncoder | None = None
        self._enc_desc_encoder: OneHotEncoder | None = None
        self._train_medians: dict[str, float] | None = None
        self._gender_cols: list[str] = []
        self._ms_cols: list[str] = []
        self._ohe_cols: list[str] = []
        self._enc_desc_cols: list[str] = []

    def fit_transform(
        self,
        patients: pd.DataFrame,
        train_encounter_ids: set | list,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Fit on train subset and transform full patients.
        Returns (patients with feature columns, list of feature column names).
        """
        train_ids = set(train_encounter_ids)
        pat = patients.drop(
            columns=[c for c in DROP_PATIENT_COLS if c in patients.columns]
        ).copy()
        patients_train = pat[pat["encounter_id"].isin(train_ids)]

        # --- Numeric: missingness, impute, scale ---
        pat["bmi_missing"] = pat["bmi"].isna().astype(float)
        pat["pain_score_missing"] = pat["pain_score"].isna().astype(float)

        self._train_medians = {
            col: patients_train[col].median() for col in NUMERIC_PAT_COLS
        }
        for col in NUMERIC_PAT_COLS:
            pat[col] = pat[col].fillna(self._train_medians[col])

        self._num_scaler = StandardScaler()
        self._num_scaler.fit(
            patients_train[NUMERIC_PAT_COLS].fillna(self._train_medians)
        )
        scaled = self._num_scaler.transform(pat[NUMERIC_PAT_COLS])
        for i, col in enumerate(NUMERIC_PAT_COLS):
            pat[f"{col}_scaled"] = scaled[:, i]

        # --- Age flags ---
        pat["is_elderly"] = (pat["age"] >= 65).astype(float)
        pat["is_child"] = (pat["age"] < 18).astype(float)

        # --- Gender OHE ---
        gender_norm = pat["gender"].replace(
            {"Male": "M", "male": "M", "Female": "F", "female": "F"}
        )
        pat["gender"] = gender_norm.fillna("unknown")

        self._gender_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self._gender_encoder.fit(
            patients_train[["gender"]].fillna("unknown")
        )
        self._gender_cols = self._gender_encoder.get_feature_names_out(
            ["gender"]
        ).tolist()
        arr = self._gender_encoder.transform(pat[["gender"]])
        for i, c in enumerate(self._gender_cols):
            pat[c] = arr[:, i]

        # --- Marital status OHE ---
        pat["marital_status"] = pat["marital_status"].fillna("unknown")
        self._ms_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self._ms_encoder.fit(
            patients_train[["marital_status"]].fillna("unknown")
        )
        self._ms_cols = self._ms_encoder.get_feature_names_out(
            ["marital_status"]
        ).tolist()
        arr = self._ms_encoder.transform(pat[["marital_status"]])
        for i, c in enumerate(self._ms_cols):
            pat[c] = arr[:, i]

        # --- Race / ethnicity OHE ---
        cat_cols = ["race", "ethnicity"]
        for col in cat_cols:
            pat[col] = pat[col].fillna("unknown")
        self._cat_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self._cat_encoder.fit(patients_train[cat_cols].fillna("unknown"))
        self._ohe_cols = self._cat_encoder.get_feature_names_out(
            cat_cols
        ).tolist()
        arr = self._cat_encoder.transform(pat[cat_cols])
        for i, c in enumerate(self._ohe_cols):
            pat[c] = arr[:, i]

        # --- Encounter description OHE ---
        pat["encounter_description"] = pat["encounter_description"].fillna(
            "unknown"
        )
        self._enc_desc_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self._enc_desc_encoder.fit(
            patients_train[["encounter_description"]].fillna("unknown")
        )
        self._enc_desc_cols = self._enc_desc_encoder.get_feature_names_out(
            ["encounter_description"]
        ).tolist()
        arr = self._enc_desc_encoder.transform(pat[["encounter_description"]])
        for i, c in enumerate(self._enc_desc_cols):
            pat[c] = arr[:, i]

        # --- Reason for visit: risk tier + missing ---
        pat["reason_risk_tier"] = (
            pat["reason_for_visit"]
            .apply(_assign_reason_risk_tier)
            .astype(float)
        )
        pat["reason_missing"] = pat["reason_for_visit"].isna().astype(float)

        # --- Comorbidities ---
        history = pat["previous_medical_history"].fillna("").str.lower()
        for feat_name, pattern in COMORBIDITY_KEYWORDS.items():
            pat[feat_name] = history.str.contains(pattern, regex=True).astype(
                float
            )
        comorbidity_flag_cols = list(COMORBIDITY_KEYWORDS.keys())
        pat["comorbidity_count"] = pat[comorbidity_flag_cols].sum(axis=1)
        pat["has_medical_history"] = (
            pat["previous_medical_history"].notna().astype(float)
        )

        # --- Medication flags ---
        meds = pat["current_medications"].fillna("").str.lower()
        pat["on_cardiac_meds"] = meds.str.contains(
            CARDIAC_MED_KEYWORDS, regex=True
        ).astype(float)
        pat["on_insulin"] = meds.str.contains(
            "insulin|humulin", regex=True
        ).astype(float)
        pat["has_medications"] = pat["current_medications"].notna().astype(
            float
        )

        # Assemble feature column list (same order as notebook)
        feature_cols = (
            [f"{c}_scaled" for c in NUMERIC_PAT_COLS]
            + ["bmi_missing", "pain_score_missing", "reason_missing"]
            + ["is_elderly", "is_child"]
            + self._gender_cols
            + self._ms_cols
            + self._ohe_cols
            + self._enc_desc_cols
            + ["reason_risk_tier"]
            + comorbidity_flag_cols
            + ["comorbidity_count", "has_medical_history"]
            + ["on_cardiac_meds", "on_insulin", "has_medications"]
        )

        return pat, feature_cols


def build_patient_features(
    patients: pd.DataFrame,
    train_encounter_ids: set | list,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build patient features with leakage-safe fitting.
    Returns (patients with feature columns, list of feature column names).
    """
    builder = PatientFeatureBuilder()
    return builder.fit_transform(patients, train_encounter_ids)


def join_patient_features(
    ts_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    pat_cols: list[str],
    on: str = "encounter_id",
) -> pd.DataFrame:
    """Left join patient feature columns onto time-series DataFrame."""
    to_join = pat_df[[on] + pat_cols].drop_duplicates(on)
    return ts_df.merge(to_join, on=on, how="left")
