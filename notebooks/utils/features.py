"""
Feature extraction utilities for vital sign time-series and patient demographics.

Single-class pipeline: FeatureEngineer handles vital imputation, lag/spectral/rolling
features, derived vitals, temporal features, ECG features, prior labels, and patient
features (EDA-informed, leakage-safe).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer:
    """
    End-to-end feature engineering pipeline. All logic is self-contained;
    methods add feature groups and return (df, list of new column names).
    """

    # Prior label feature column names
    PRIOR_LABEL_COLS = [
        "prior_label",
        "max_label_last_60s",
        "max_label_encounter",
        "ever_deteriorated",
    ]

    # ECG feature column names
    ECG_COLS = [
        "ecg_hr_bpm",
        "ecg_mean",
        "ecg_std",
        "ecg_min",
        "ecg_max",
        "ecg_skew",
        "ecg_kurtosis",
        "ecg_range",
        "ecg_peaks",
        "ecg_dom_freq",
        "ecg_power_lf",
        "ecg_power_hf",
        "ecg_lf_hf_ratio",
        "ecg_spectral_entropy",
        "ecg_total_power",
        "hr_ecg_diff",
    ]

    # Standard HRV frequency bands (Hz) per Task Force guidelines
    ECG_LF_BAND = (0.04, 0.15)
    ECG_HF_BAND = (0.15, 0.4)
    ECG_DEFAULT_FS = 250.0

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

    def __init__(
        self,
        vital_cols: list[str] | None = None,
        n_lags: int = 12,
        rolling_windows: list[int] | None = None,
        data_dir: Path | str | None = None,
        encounter_id_col: str = "encounter_id",
    ) -> None:
        self.vital_cols = vital_cols or [
            "heart_rate",
            "systolic_bp",
            "diastolic_bp",
            "respiratory_rate",
            "oxygen_saturation",
        ]
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows or [6, 12, 24, 60]
        self.data_dir = Path(data_dir) if data_dir else None
        self.encounter_id_col = encounter_id_col

    def impute_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Neighbour interpolation + encounter median fallback."""
        out = df.sort_values([self.encounter_id_col, "timestamp"]).reset_index(
            drop=True
        ).copy()
        for col in self.vital_cols:
            prev = out.groupby(self.encounter_id_col)[col].shift(1)
            nxt = out.groupby(self.encounter_id_col)[col].shift(-1)
            neighbour_mean = pd.concat([prev, nxt], axis=1).mean(axis=1)
            out[col] = out[col].fillna(neighbour_mean)
            out[col] = out.groupby(self.encounter_id_col)[col].transform(
                lambda x: x.fillna(x.median())
            )
        return out

    def add_lag_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Lag1..lag{n_lags} per vital.

        Warmup NaNs are backfilled with the encounter's first observation
        (not median).  This ensures derivative features are exactly 0 during
        warmup ("no change detected yet") rather than noisy deviations from
        the encounter median.

        Also adds ``warmup_progress`` (0 → 1 over the first ``n_lags`` rows)
        so the model can learn to discount derivative/rolling features when
        the lag window is still filling.
        """
        out = df.sort_values(
            [self.encounter_id_col, "timestamp"]
        ).reset_index(drop=True).copy()
        for col in self.vital_cols:
            g = out.groupby(self.encounter_id_col)[col]
            for lag in range(1, self.n_lags + 1):
                out[f"{col}_lag{lag}"] = g.shift(lag)

        # Backfill: use encounter's first observation (not median)
        for col in self.vital_cols:
            first_val = out.groupby(self.encounter_id_col)[col].transform(
                "first"
            )
            for lag in range(1, self.n_lags + 1):
                lag_col = f"{col}_lag{lag}"
                out[lag_col] = out[lag_col].fillna(first_val)

        # warmup_progress: fraction of the lag window with real observations
        # Row 0 → 0/n_lags, Row 1 → 1/n_lags, ..., Row n_lags → 1.0
        row_in_enc = out.groupby(self.encounter_id_col).cumcount()
        out["warmup_progress"] = (
            row_in_enc.clip(upper=self.n_lags) / self.n_lags
        ).astype(np.float32)

        lag_cols = [
            f"{col}_lag{lag}"
            for col in self.vital_cols
            for lag in range(1, self.n_lags + 1)
        ]
        return out, lag_cols + ["warmup_progress"]

    def add_spectral_features(
        self,
        df: pd.DataFrame,
        vital_cols: list[str] | None = None,
        n_window: int = 37,
        fs: float = 0.2,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Add spectral features (dom_freq, power_low, autocorr) per vital."""
        vcols = vital_cols or self.vital_cols
        out = df.copy()
        new_cols: list[str] = []
        suffix = f"_w{n_window}"

        for col in vcols:
            if col not in out.columns:
                raise ValueError(f"Column '{col}' not in dataframe")
            # Build rolling window
            shifts = list(range(n_window - 1, -1, -1))
            stacked = np.column_stack(
                [
                    out.groupby(self.encounter_id_col)[col]
                    .shift(s)
                    .values
                    for s in shifts
                ]
            )
            window = stacked.astype(np.float64)
            # Impute window NaNs with encounter's first value (not median)
            # to avoid artificial variance in warmup rows
            encounter_first = (
                out.groupby(self.encounter_id_col)[col]
                .transform("first")
                .values
            )
            window = np.where(
                np.isnan(window), encounter_first[:, np.newaxis], window
            )
            # Compute spectral features
            n = window.shape[1]
            if n < 4:
                nan_arr = np.full(window.shape[0], np.nan, dtype=np.float64)
                dom_freq, power_low, autocorr = nan_arr, nan_arr, nan_arr
            else:
                row_median = np.nanmedian(window, axis=1, keepdims=True)
                x = np.where(np.isnan(window), row_median, window)
                x_mean = x.mean(axis=1, keepdims=True)
                x_detrend = x - x_mean
                hanning = np.hanning(n)
                x_windowed = x_detrend * hanning
                fft_vals = np.fft.rfft(x_windowed, axis=1)
                freqs = np.fft.rfftfreq(n, 1.0 / fs)
                power = np.abs(fft_vals) ** 2
                power[:, 0] = 0
                total_power = power.sum(axis=1)
                valid_power = total_power >= 1e-10
                power_ac = power[:, 1:]
                dom_idx = np.argmax(power_ac, axis=1)
                dom_freq = freqs[1:][dom_idx].astype(np.float64)
                dom_freq = np.where(valid_power, dom_freq, np.nan)
                low_mask = (freqs > 0) & (freqs <= 0.05)
                power_low = np.sum(power[:, low_mask], axis=1) / np.where(
                    valid_power, total_power, np.nan
                )
                power_low = np.where(valid_power, power_low, np.nan)
                x_centered = x - x_mean
                x0, x1 = x_centered[:, :-1], x_centered[:, 1:]
                cov = np.sum(x0 * x1, axis=1)
                var_product = np.sum(x0**2, axis=1) * np.sum(x1**2, axis=1)
                valid_autocorr = var_product >= 1e-20
                autocorr = np.where(
                    valid_autocorr,
                    cov / np.sqrt(var_product),
                    np.nan,
                )
            out[f"{col}_dom_freq{suffix}"] = dom_freq
            out[f"{col}_power_low{suffix}"] = power_low
            out[f"{col}_autocorr{suffix}"] = autocorr
            new_cols.extend(
                [
                    f"{col}_dom_freq{suffix}",
                    f"{col}_power_low{suffix}",
                    f"{col}_autocorr{suffix}",
                ]
            )
        return out, new_cols

    def add_derivative_features(
        self, df: pd.DataFrame, lag_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """delta, delta_1s, accel per vital."""
        out = df.copy()
        for v in self.vital_cols:
            out[f"{v}_delta"] = out[v] - out[f"{v}_lag{self.n_lags}"]
            out[f"{v}_delta_1s"] = out[v] - out[f"{v}_lag1"]
            out[f"{v}_accel"] = (
                out[v] - 2 * out[f"{v}_lag1"] + out[f"{v}_lag2"]
            )
        cols = [
            f"{v}{s}"
            for v in self.vital_cols
            for s in ("_delta", "_delta_1s", "_accel")
        ]
        return out, cols

    def add_rolling_stats(
        self, df: pd.DataFrame, lag_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """mean, std, min, max over [lag12..current] per vital."""
        out = df.copy()
        for v in self.vital_cols:
            window_cols = [
                f"{v}_lag{i}"
                for i in range(self.n_lags, 0, -1)
            ] + [v]
            window = out[window_cols]
            out[f"{v}_mean"] = window.mean(axis=1)
            out[f"{v}_std"] = window.std(axis=1, ddof=0)
            out[f"{v}_min"] = window.min(axis=1)
            out[f"{v}_max"] = window.max(axis=1)
        cols = [
            f"{v}{s}"
            for v in self.vital_cols
            for s in ("_mean", "_std", "_min", "_max")
        ]
        return out, cols

    def add_multiscale_rolling_stats(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Rolling mean/std/min/max per vital per window."""
        out = df.sort_values(
            [self.encounter_id_col, "timestamp"]
        ).reset_index(drop=True).copy()
        cols: list[str] = []
        for v in self.vital_cols:
            g = out.groupby(self.encounter_id_col)[v]
            for w in self.rolling_windows:
                out[f"{v}_roll_mean_{w}"] = g.transform(
                    lambda x, win=w: x.rolling(window=win, min_periods=1).mean()
                )
                out[f"{v}_roll_std_{w}"] = g.transform(
                    lambda x, win=w: x.rolling(
                        window=win, min_periods=1
                    ).std().fillna(0)
                )
                out[f"{v}_roll_min_{w}"] = g.transform(
                    lambda x, win=w: x.rolling(window=win, min_periods=1).min()
                )
                out[f"{v}_roll_max_{w}"] = g.transform(
                    lambda x, win=w: x.rolling(window=win, min_periods=1).max()
                )
                cols.extend([
                    f"{v}_roll_mean_{w}",
                    f"{v}_roll_std_{w}",
                    f"{v}_roll_min_{w}",
                    f"{v}_roll_max_{w}",
                ])
        return out, cols

    def add_derived_vitals(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """pulse_pressure, map, deltas, shock_index, hr_rr_ratio."""
        out = df.copy()
        out["pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
        out["map"] = (out["systolic_bp"] + 2 * out["diastolic_bp"]) / 3
        pp_lag = (
            out[f"systolic_bp_lag{self.n_lags}"]
            - out[f"diastolic_bp_lag{self.n_lags}"]
        )
        map_lag = (
            out[f"systolic_bp_lag{self.n_lags}"]
            + 2 * out[f"diastolic_bp_lag{self.n_lags}"]
        ) / 3
        out["pulse_pressure_delta"] = out["pulse_pressure"] - pp_lag
        out["map_delta"] = out["map"] - map_lag
        out["shock_index"] = (
            out["heart_rate"] / out["systolic_bp"].replace(0, np.nan)
        ).fillna(0)
        out["hr_rr_ratio"] = (
            out["heart_rate"] / out["respiratory_rate"].replace(0, np.nan)
        ).fillna(0)
        cols = [
            "pulse_pressure",
            "map",
            "pulse_pressure_delta",
            "map_delta",
            "shock_index",
            "hr_rr_ratio",
        ]
        return out, cols

    def add_temporal_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """minutes_into_encounter, hour_sin/cos, dow_sin/cos."""
        out = df.copy()
        enc_start = out.groupby(self.encounter_id_col)["timestamp"].transform(
            "min"
        )
        out["minutes_into_encounter"] = (
            (out["timestamp"] - enc_start).dt.total_seconds() / 60
        )
        hour = out["timestamp"].dt.hour
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = out["timestamp"].dt.dayofweek
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        cols = [
            "minutes_into_encounter",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
        return out, cols

    def add_ecg_features(
        self, df: pd.DataFrame, split: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """Load ECG from {data_dir}/{split}_ecg.npz, merge, or zeros."""
        ecg_path = (
            self.data_dir / f"{split}_ecg.npz"
            if self.data_dir
            else None
        )
        if ecg_path is not None and ecg_path.exists():
            data = np.load(ecg_path, allow_pickle=True)
            encounter_ids = data["encounter_ids"]
            ecg_data = data["ecg_data"]
            hr_bpm = data["hr_bpm"]
            fs = (
                float(data["sample_rate"])
                if "sample_rate" in data
                else self.ECG_DEFAULT_FS
            )
            records = []
            for i, eid in enumerate(encounter_ids):
                signal = np.asarray(ecg_data[i], dtype=np.float64)
                spectral = self._ecg_spectral_features(signal, fs)
                records.append({
                    "encounter_id": eid,
                    "ecg_hr_bpm": float(hr_bpm[i]),
                    "ecg_mean": float(signal.mean()),
                    "ecg_std": float(signal.std()),
                    "ecg_min": float(signal.min()),
                    "ecg_max": float(signal.max()),
                    "ecg_skew": float(stats.skew(signal)),
                    "ecg_kurtosis": float(stats.kurtosis(signal)),
                    "ecg_range": float(signal.max() - signal.min()),
                    "ecg_peaks": int(
                        np.sum(
                            (signal[1:-1] > signal[:-2])
                            & (signal[1:-1] > signal[2:])
                        )
                    ),
                    **spectral,
                })
            ecg_feats = pd.DataFrame(records)
            out = df.merge(ecg_feats, on="encounter_id", how="left")
            out["hr_ecg_diff"] = out["heart_rate"] - out["ecg_hr_bpm"]
            for c in self.ECG_COLS:
                if c in out.columns:
                    out[c] = out[c].fillna(0)
        else:
            out = df.copy()
            n = len(out)
            for c in self.ECG_COLS:
                out[c] = np.zeros(n)
        return out, list(self.ECG_COLS)

    def _ecg_spectral_features(
        self, signal: np.ndarray, fs: float
    ) -> dict:
        """FFT-based ECG spectral features per signal."""
        n = len(signal)
        out = {
            "ecg_dom_freq": np.nan,
            "ecg_power_lf": np.nan,
            "ecg_power_hf": np.nan,
            "ecg_lf_hf_ratio": np.nan,
            "ecg_spectral_entropy": np.nan,
            "ecg_total_power": np.nan,
        }
        if n < 8:
            return out
        x = np.asarray(signal, dtype=np.float64)
        x = np.nan_to_num(x, nan=np.nanmean(x), posinf=0, neginf=0)
        x = x - np.mean(x)
        x_windowed = x * np.hanning(n)
        fft_vals = np.fft.rfft(x_windowed)
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        power = np.abs(fft_vals) ** 2
        power[0] = 0
        total_power = np.sum(power)
        if total_power < 1e-12:
            return out
        out["ecg_total_power"] = float(total_power)
        idx_ac = np.argmax(power[1:]) + 1
        out["ecg_dom_freq"] = float(freqs[idx_ac])
        lf_mask = (
            (freqs >= self.ECG_LF_BAND[0])
            & (freqs <= self.ECG_LF_BAND[1])
        )
        hf_mask = (
            (freqs >= self.ECG_HF_BAND[0])
            & (freqs <= self.ECG_HF_BAND[1])
        )
        power_lf = float(np.sum(power[lf_mask]))
        power_hf = float(np.sum(power[hf_mask]))
        out["ecg_power_lf"] = (
            power_lf / total_power if total_power > 0 else np.nan
        )
        out["ecg_power_hf"] = (
            power_hf / total_power if total_power > 0 else np.nan
        )
        out["ecg_lf_hf_ratio"] = (
            power_lf / power_hf if power_hf > 1e-12 else 0.0
        )
        p_norm = power[1:] / (np.sum(power[1:]) + 1e-12)
        p_norm = p_norm[p_norm > 0]
        out["ecg_spectral_entropy"] = float(
            -np.sum(p_norm * np.log2(p_norm + 1e-12))
        )
        return out

    def add_prior_label_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """prior_label, max_label_last_60s, max_label_encounter, ever_deteriorated."""
        out = df.sort_values(
            [self.encounter_id_col, "timestamp"]
        ).reset_index(drop=True).copy()
        if "label" not in df.columns:
            for col in self.PRIOR_LABEL_COLS:
                out[col] = 0.0
            return out, list(self.PRIOR_LABEL_COLS)
        out["prior_label"] = out.groupby(self.encounter_id_col)["label"].shift(
            1
        ).fillna(0)
        out["max_label_last_60s"] = (
            out.groupby(self.encounter_id_col)["label"]
            .transform(
                lambda x: x.shift(1).rolling(
                    self.n_lags, min_periods=1
                ).max()
            )
            .fillna(0)
        )
        out["max_label_encounter"] = (
            out.groupby(self.encounter_id_col)["label"]
            .transform(lambda x: x.shift(1).expanding().max())
            .fillna(0)
        )
        out["ever_deteriorated"] = (out["max_label_encounter"] > 0).astype(
            float
        )
        return out, list(self.PRIOR_LABEL_COLS)

    # ------------------------------------------------------------------
    # Clinical alert features
    # ------------------------------------------------------------------

    def add_clinical_alert_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Binary flags for clinically dangerous vital sign ranges."""
        out = df.copy()
        out["hr_tachycardia"] = (out["heart_rate"] > 100).astype(float)
        out["hr_bradycardia"] = (out["heart_rate"] < 60).astype(float)
        out["hr_critical_high"] = (out["heart_rate"] > 150).astype(float)
        out["bp_hypotension"] = (out["systolic_bp"] < 90).astype(float)
        out["bp_hypertension"] = (out["systolic_bp"] > 180).astype(float)
        out["spo2_low"] = (out["oxygen_saturation"] < 92).astype(float)
        out["spo2_critical"] = (out["oxygen_saturation"] < 88).astype(float)
        out["rr_tachypnea"] = (out["respiratory_rate"] > 22).astype(float)
        out["rr_bradypnea"] = (out["respiratory_rate"] < 8).astype(float)
        alert_flags = [
            "hr_tachycardia", "hr_bradycardia", "bp_hypotension",
            "spo2_low", "rr_tachypnea", "rr_bradypnea",
        ]
        out["n_active_alerts"] = out[alert_flags].sum(axis=1)
        cols = [
            "hr_tachycardia", "hr_bradycardia", "hr_critical_high",
            "bp_hypotension", "bp_hypertension",
            "spo2_low", "spo2_critical",
            "rr_tachypnea", "rr_bradypnea",
            "n_active_alerts",
        ]
        return out, cols

    # ------------------------------------------------------------------
    # Interaction features (vital x demographic)
    # ------------------------------------------------------------------

    def add_interaction_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Cross vital signs with patient demographics."""
        out = df.copy()
        cols: list[str] = []
        if "is_elderly" in out.columns:
            for v in self.vital_cols:
                c = f"{v}_x_elderly"
                out[c] = out[v] * out["is_elderly"]
                cols.append(c)
        if "is_child" in out.columns:
            for v in self.vital_cols:
                c = f"{v}_x_child"
                out[c] = out[v] * out["is_child"]
                cols.append(c)
        if "comorbidity_count" in out.columns:
            for v in ["heart_rate", "oxygen_saturation", "systolic_bp"]:
                c = f"{v}_x_comorbidity"
                out[c] = out[v] * out["comorbidity_count"]
                cols.append(c)
        if "on_cardiac_meds" in out.columns:
            out["hr_x_cardiac_meds"] = out["heart_rate"] * out["on_cardiac_meds"]
            cols.append("hr_x_cardiac_meds")
        if "reason_risk_tier" in out.columns and "shock_index" in out.columns:
            out["shock_x_risk"] = out["shock_index"] * out["reason_risk_tier"]
            cols.append("shock_x_risk")
        return out, cols

    # ------------------------------------------------------------------
    # Higher-order derivatives
    # ------------------------------------------------------------------

    def add_higher_order_derivatives(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Jerk (3rd derivative) for sudden vital changes."""
        out = df.copy()
        cols: list[str] = []
        for v in self.vital_cols:
            if f"{v}_lag3" in out.columns:
                c = f"{v}_jerk"
                out[c] = (
                    out[v]
                    - 3 * out[f"{v}_lag1"]
                    + 3 * out[f"{v}_lag2"]
                    - out[f"{v}_lag3"]
                )
                cols.append(c)
        return out, cols

    def add_patient_features(
        self,
        df: pd.DataFrame,
        patients: pd.DataFrame,
        train_encounter_ids: set | list,
    ) -> tuple[pd.DataFrame, list[str]]:
        """EDA-informed patient features (leakage-safe)."""
        train_ids = set(train_encounter_ids)
        pat = patients.drop(
            columns=[
                c
                for c in self.DROP_PATIENT_COLS
                if c in patients.columns
            ]
        ).copy()
        patients_train = pat[pat["encounter_id"].isin(train_ids)]

        pat["bmi_missing"] = pat["bmi"].isna().astype(float)
        pat["pain_score_missing"] = pat["pain_score"].isna().astype(float)
        train_medians = {
            col: patients_train[col].median()
            for col in self.NUMERIC_PAT_COLS
        }
        for col in self.NUMERIC_PAT_COLS:
            pat[col] = pat[col].fillna(train_medians[col])
        num_scaler = StandardScaler()
        num_scaler.fit(
            patients_train[self.NUMERIC_PAT_COLS].fillna(train_medians)
        )
        scaled = num_scaler.transform(pat[self.NUMERIC_PAT_COLS])
        for i, col in enumerate(self.NUMERIC_PAT_COLS):
            pat[f"{col}_scaled"] = scaled[:, i]

        pat["is_elderly"] = (pat["age"] >= 65).astype(float)
        pat["is_child"] = (pat["age"] < 18).astype(float)

        gender_norm = pat["gender"].replace(
            {"Male": "M", "male": "M", "Female": "F", "female": "F"}
        )
        pat["gender"] = gender_norm.fillna("unknown")
        gender_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        gender_encoder.fit(
            patients_train[["gender"]].fillna("unknown")
        )
        gender_cols = gender_encoder.get_feature_names_out(["gender"]).tolist()
        arr = gender_encoder.transform(pat[["gender"]])
        for i, c in enumerate(gender_cols):
            pat[c] = arr[:, i]

        pat["marital_status"] = pat["marital_status"].fillna("unknown")
        ms_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        ms_encoder.fit(
            patients_train[["marital_status"]].fillna("unknown")
        )
        ms_cols = ms_encoder.get_feature_names_out(
            ["marital_status"]
        ).tolist()
        arr = ms_encoder.transform(pat[["marital_status"]])
        for i, c in enumerate(ms_cols):
            pat[c] = arr[:, i]

        cat_cols = ["race", "ethnicity"]
        for col in cat_cols:
            pat[col] = pat[col].fillna("unknown")
        cat_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        cat_encoder.fit(
            patients_train[cat_cols].fillna("unknown")
        )
        ohe_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
        arr = cat_encoder.transform(pat[cat_cols])
        for i, c in enumerate(ohe_cols):
            pat[c] = arr[:, i]

        pat["encounter_description"] = pat["encounter_description"].fillna(
            "unknown"
        )
        enc_desc_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        enc_desc_encoder.fit(
            patients_train[["encounter_description"]].fillna("unknown")
        )
        enc_desc_cols = enc_desc_encoder.get_feature_names_out(
            ["encounter_description"]
        ).tolist()
        arr = enc_desc_encoder.transform(pat[["encounter_description"]])
        for i, c in enumerate(enc_desc_cols):
            pat[c] = arr[:, i]

        def _risk_tier(reason: str | float) -> int:
            if pd.isna(reason):
                return 1
            r = str(reason).lower()
            for hr in self.HIGH_RISK_REASONS:
                if hr.lower() in r:
                    return 2
            for mr in self.MEDIUM_RISK_REASONS:
                if mr.lower() in r:
                    return 1
            return 0

        pat["reason_risk_tier"] = (
            pat["reason_for_visit"].apply(_risk_tier).astype(float)
        )
        pat["reason_missing"] = pat["reason_for_visit"].isna().astype(float)

        history = pat["previous_medical_history"].fillna("").str.lower()
        for feat_name, pattern in self.COMORBIDITY_KEYWORDS.items():
            pat[feat_name] = history.str.contains(
                pattern, regex=True
            ).astype(float)
        comorbidity_flag_cols = list(self.COMORBIDITY_KEYWORDS.keys())
        pat["comorbidity_count"] = pat[comorbidity_flag_cols].sum(axis=1)
        pat["has_medical_history"] = (
            pat["previous_medical_history"].notna().astype(float)
        )

        meds = pat["current_medications"].fillna("").str.lower()
        pat["on_cardiac_meds"] = meds.str.contains(
            self.CARDIAC_MED_KEYWORDS, regex=True
        ).astype(float)
        pat["on_insulin"] = meds.str.contains(
            "insulin|humulin", regex=True
        ).astype(float)
        pat["has_medications"] = pat["current_medications"].notna().astype(
            float
        )

        pat_cols = (
            [f"{c}_scaled" for c in self.NUMERIC_PAT_COLS]
            + ["bmi_missing", "pain_score_missing", "reason_missing"]
            + ["is_elderly", "is_child"]
            + gender_cols
            + ms_cols
            + ohe_cols
            + enc_desc_cols
            + ["reason_risk_tier"]
            + comorbidity_flag_cols
            + ["comorbidity_count", "has_medical_history"]
            + ["on_cardiac_meds", "on_insulin", "has_medications"]
        )

        to_join = pat[[self.encounter_id_col] + pat_cols].drop_duplicates(
            self.encounter_id_col
        )
        out = df.merge(to_join, on=self.encounter_id_col, how="left")
        return out, pat_cols

    def transform(
        self,
        train_raw: pd.DataFrame,
        test_raw: pd.DataFrame,
        holdout_raw: pd.DataFrame,
        patients: pd.DataFrame,
        include_prior_labels: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Full pipeline: impute → lag → derivative → rolling → multiscale rolling
        → derived vitals → temporal → ECG → (optional) prior labels → patient
        features → clinical alerts → interactions → higher-order derivatives.

        Args:
            include_prior_labels: If ``False`` (default), prior-label features
                are **not** generated.  These leak label information and are
                all-zero on holdout, causing the model to predict everything
                as class 0.

        Returns (train, test, holdout, feature_cols).
        """
        logger.info("Creating features: impute_vitals (neighbour + median fallback)")
        train = self.impute_vitals(train_raw)
        test = self.impute_vitals(test_raw)
        holdout = self.impute_vitals(holdout_raw)

        logger.info("Creating features: lag (vital lags 1..n + warmup_progress)")
        train, lag_cols = self.add_lag_features(train)
        test, _ = self.add_lag_features(test)
        holdout, _ = self.add_lag_features(holdout)

        logger.info("Creating features: derivative (delta, delta_1s, accel per vital)")
        train, deriv_cols = self.add_derivative_features(train, lag_cols)
        test, _ = self.add_derivative_features(test, lag_cols)
        holdout, _ = self.add_derivative_features(holdout, lag_cols)

        logger.info("Creating features: rolling_stats (mean/std/min/max over lag window)")
        train, rolling_cols = self.add_rolling_stats(train, lag_cols)
        test, _ = self.add_rolling_stats(test, lag_cols)
        holdout, _ = self.add_rolling_stats(holdout, lag_cols)

        logger.info("Creating features: multiscale_rolling (mean/std/min/max per window)")
        train, multi_rolling_cols = self.add_multiscale_rolling_stats(train)
        test, _ = self.add_multiscale_rolling_stats(test)
        holdout, _ = self.add_multiscale_rolling_stats(holdout)

        logger.info("Creating features: derived_vitals (pulse_pressure, map, shock_index, hr_rr_ratio)")
        train, derived_cols = self.add_derived_vitals(train)
        test, _ = self.add_derived_vitals(test)
        holdout, _ = self.add_derived_vitals(holdout)

        logger.info("Creating features: temporal (minutes_into_encounter, hour/dow sin/cos)")
        train, temporal_cols = self.add_temporal_features(train)
        test, _ = self.add_temporal_features(test)
        holdout, _ = self.add_temporal_features(holdout)

        logger.info("Creating features: ecg (stats + FFT: dom_freq, LF/HF, spectral_entropy, hr_ecg_diff)")
        train, ecg_cols = self.add_ecg_features(train, "train")
        test, _ = self.add_ecg_features(test, "test")
        holdout, _ = self.add_ecg_features(holdout, "holdout")

        if include_prior_labels:
            logger.info("Creating features: prior_label (prior_label, max_label_last_60s, ever_deteriorated)")
            train, prior_cols = self.add_prior_label_features(train)
            test, _ = self.add_prior_label_features(test)
            holdout, _ = self.add_prior_label_features(holdout)
        else:
            prior_cols = []

        logger.info("Creating features: patient (demographics, comorbidity, risk tier, OHE)")
        train_ids = set(train[self.encounter_id_col].unique())
        train, patient_cols = self.add_patient_features(
            train, patients, train_ids
        )
        test, _ = self.add_patient_features(test, patients, train_ids)
        holdout, _ = self.add_patient_features(holdout, patients, train_ids)

        logger.info("Creating features: clinical_alert (tachycardia, hypotension, spo2_low, n_active_alerts)")
        train, alert_cols = self.add_clinical_alert_features(train)
        test, _ = self.add_clinical_alert_features(test)
        holdout, _ = self.add_clinical_alert_features(holdout)

        logger.info("Creating features: interaction (vital x elderly/child/comorbidity/cardiac_meds)")
        train, interaction_cols = self.add_interaction_features(train)
        test, _ = self.add_interaction_features(test)
        holdout, _ = self.add_interaction_features(holdout)

        logger.info("Creating features: higher_order_derivatives (jerk per vital)")
        train, higher_deriv_cols = self.add_higher_order_derivatives(train)
        test, _ = self.add_higher_order_derivatives(test)
        holdout, _ = self.add_higher_order_derivatives(holdout)

        feature_cols = (
            self.vital_cols
            + lag_cols
            + deriv_cols
            + rolling_cols
            + multi_rolling_cols
            + derived_cols
            + temporal_cols
            + ecg_cols
            + prior_cols
            + patient_cols
            + alert_cols
            + interaction_cols
            + higher_deriv_cols
        )
        return train, test, holdout, feature_cols
