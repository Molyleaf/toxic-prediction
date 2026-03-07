from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


DEFAULT_DATA_FILE = "质谱数据汇总_处理后后后2.csv"
DEFAULT_TARGET_COLUMN = "毒性"
RETENTION_TIME_COLUMN = "RETENTION_TIME"
RETENTION_TIME_NUMBER_PATTERN = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")
LIGHTGBM_GPU_PIP_INSTALL_COMMAND = (
    "pip install lightgbm --no-binary lightgbm "
    "--config-settings=cmake.define.USE_GPU=ON"
)
LIGHTGBM_CUDA_PIP_INSTALL_COMMAND = (
    "pip install lightgbm --no-binary lightgbm "
    "--config-settings=cmake.define.USE_CUDA=ON"
)
LIGHTGBM_GPU_SOURCE_BUILD_COMMANDS = (
    "git clone --recursive https://github.com/microsoft/LightGBM\n"
    "cd LightGBM\n"
    "cmake -B build -S . -DUSE_GPU=ON\n"
    "cmake --build build -j4"
)
LIGHTGBM_CUDA_SOURCE_BUILD_COMMANDS = (
    "git clone --recursive https://github.com/microsoft/LightGBM\n"
    "cd LightGBM\n"
    "cmake -B build -S . -DUSE_CUDA=ON\n"
    "cmake --build build -j4"
)
DEFAULT_SMOTE_SAMPLING_STRATEGY = 0.75
DEFAULT_INITIAL_THRESHOLD = 0.42
DEFAULT_CALIBRATION_METHOD = "isotonic"
DEFAULT_BORDERLINE_SMOTE_KIND = "borderline-1"
DEFAULT_LGBM_DEVICE_TYPE = "gpu"
DEFAULT_LGBM_MAX_BIN = 255
DEFAULT_LGBM_GPU_MAX_BIN = 63
ARTIFACTS_VERSION = 3


@dataclass(frozen=True)
class NotebookRunConfig:
    bayes_n_iter: int
    cv_folds: int
    random_seed: int
    test_size: float
    device_type: str
    model_n_jobs: int
    search_n_jobs: int
    smote_k_neighbors: int
    smote_sampling_strategy: float
    early_stopping_rounds: int
    early_stopping_validation_fraction: float
    calibration_method: str
    initial_threshold: float


def format_notebook_run_summary(
    config: NotebookRunConfig,
    data_path: Optional[Path] = None,
) -> str:
    lines = [
        f"随机种子: {config.random_seed}",
        f"测试集比例: {config.test_size}",
        f"训练设备: {config.device_type}",
        f"BayesSearch n_iter: {config.bayes_n_iter}",
        f"CV folds: {config.cv_folds}",
        f"模型 n_jobs: {config.model_n_jobs}",
        f"搜索 n_jobs: {config.search_n_jobs}",
        f"SMOTE k_neighbors: {config.smote_k_neighbors}",
        f"Borderline-SMOTE 目标少数类/多数类比例: {config.smote_sampling_strategy}",
        f"折内 early stopping rounds: {config.early_stopping_rounds}",
        f"折内 early stopping 验证集比例: {config.early_stopping_validation_fraction}",
        f"概率校准方法: {config.calibration_method}",
        f"初始阈值: {config.initial_threshold}",
    ]
    if data_path is not None:
        lines.insert(1, f"数据文件: {data_path}")
    return "\n".join(lines)


def resolve_recommended_model_n_jobs(explicit_n_jobs: Optional[int] = None) -> int:
    if explicit_n_jobs is not None:
        resolved = int(explicit_n_jobs)
        if resolved <= 0:
            raise ValueError("model_n_jobs 必须为正整数。")
        return resolved

    cpu_count = os.cpu_count() or 8
    # LightGBM 的 GPU / CUDA 训练仍依赖 CPU 侧做直方图构建与数据调度；
    # 默认给到 4~16 个线程，避免 4090 被单线程喂数拖慢。
    return int(max(4, min(16, cpu_count)))


def normalize_lightgbm_device_type(device_type: Optional[str] = None) -> str:
    resolved = str(DEFAULT_LGBM_DEVICE_TYPE if device_type is None else device_type).strip().lower()
    if resolved not in {"gpu", "cuda"}:
        raise ValueError("device_type 只支持 'gpu' 或 'cuda'，项目已禁用 CPU fallback。")
    return resolved


def resolve_lightgbm_max_bin(device_type: Optional[str] = None) -> int:
    resolved_device_type = normalize_lightgbm_device_type(device_type)
    if resolved_device_type == "gpu":
        return int(DEFAULT_LGBM_GPU_MAX_BIN)
    return int(DEFAULT_LGBM_MAX_BIN)


def normalize_probability_calibration_method(method: Optional[str]) -> str:
    resolved = str(DEFAULT_CALIBRATION_METHOD if method is None else method).strip().lower()
    alias_map = {
        "sigmoid": "platt",
        "logistic": "platt",
    }
    resolved = alias_map.get(resolved, resolved)
    if resolved not in {"isotonic", "platt", "auto"}:
        raise ValueError("calibration_method 只支持 'isotonic'、'platt' 或 'auto'。")
    return resolved


def get_lightgbm_cuda_installation_notes() -> str:
    system_name = platform.system()
    lines = []

    if system_name == "Windows":
        lines.append("检测到当前系统为 Windows。官方当前不支持 Windows 上的 CUDA 版 LightGBM。")
        lines.append("如果要继续使用 device_type='cuda'，请改到 Linux 或 WSL2 训练。")
    elif system_name == "Linux":
        lines.append("检测到当前系统为 Linux。请确认 NVIDIA CUDA 环境已可用，再继续安装 LightGBM。")
    else:
        lines.append(f"检测到当前系统为 {system_name}。官方仅支持在 Linux 上使用 LightGBM 的 CUDA 版本。")

    lines.extend(
        [
            "",
            "Python 包源码安装命令:",
            f"  {LIGHTGBM_CUDA_PIP_INSTALL_COMMAND}",
            "",
            "如需先手工编译 LightGBM，可使用:",
            LIGHTGBM_CUDA_SOURCE_BUILD_COMMANDS,
        ]
    )
    return "\n".join(lines)


def get_lightgbm_gpu_installation_notes() -> str:
    system_name = platform.system()
    lines = []

    if system_name == "Windows":
        lines.append("检测到当前系统为 Windows。LightGBM 的 OpenCL GPU 路径可在 Windows 上使用。")
        lines.append("请确认显卡驱动和 OpenCL runtime 已可用，再继续安装 GPU 版 LightGBM。")
    elif system_name == "Linux":
        lines.append("检测到当前系统为 Linux。请确认 OpenCL runtime 与显卡驱动已可用。")
    else:
        lines.append(
            f"检测到当前系统为 {system_name}。请先确认该系统具备可用的 OpenCL runtime。"
        )

    lines.extend(
        [
            "",
            "Python 包源码安装命令:",
            f"  {LIGHTGBM_GPU_PIP_INSTALL_COMMAND}",
            "",
            "如需先手工编译 LightGBM，可使用:",
            LIGHTGBM_GPU_SOURCE_BUILD_COMMANDS,
        ]
    )
    return "\n".join(lines)


def get_lightgbm_device_installation_notes(device_type: Optional[str] = None) -> str:
    resolved_device_type = normalize_lightgbm_device_type(device_type)
    if resolved_device_type == "gpu":
        return get_lightgbm_gpu_installation_notes()
    return get_lightgbm_cuda_installation_notes()


def resolve_data_path(
    data_file_name: str = DEFAULT_DATA_FILE,
    start_dir: Optional[Path] = None,
) -> Path:
    base_dir = (start_dir or Path.cwd()).resolve()
    candidates = [
        base_dir / data_file_name,
        base_dir / "lightgbm" / data_file_name,
        base_dir.parent / data_file_name,
        base_dir.parent / "lightgbm" / data_file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到数据文件: {data_file_name}")


def build_notebook_run_config(
    bayes_n_iter: Optional[int] = None,
    cv_folds: Optional[int] = None,
    random_seed: Optional[int] = None,
    test_size: Optional[float] = None,
    device_type: Optional[str] = None,
    model_n_jobs: Optional[int] = None,
    search_n_jobs: Optional[int] = None,
    smote_k_neighbors: Optional[int] = None,
    smote_sampling_strategy: Optional[float] = None,
    early_stopping_rounds: Optional[int] = None,
    early_stopping_validation_fraction: Optional[float] = None,
    calibration_method: Optional[str] = None,
    initial_threshold: Optional[float] = None,
) -> NotebookRunConfig:
    resolved_random_seed = int(42 if random_seed is None else random_seed)
    resolved_test_size = float(0.2 if test_size is None else test_size)
    resolved_bayes_n_iter = int(48 if bayes_n_iter is None else bayes_n_iter)
    resolved_cv_folds = int(5 if cv_folds is None else cv_folds)
    resolved_device_type = normalize_lightgbm_device_type(device_type)

    # GPU 搜索仍保持串行，避免多个 worker 同时争抢同一张卡。
    resolved_model_n_jobs = resolve_recommended_model_n_jobs(model_n_jobs)
    resolved_search_n_jobs = int(1 if search_n_jobs is None else search_n_jobs)
    resolved_smote_k_neighbors = int(5 if smote_k_neighbors is None else smote_k_neighbors)
    resolved_smote_sampling_strategy = float(
        DEFAULT_SMOTE_SAMPLING_STRATEGY
        if smote_sampling_strategy is None
        else smote_sampling_strategy
    )
    resolved_early_stopping_rounds = int(
        300 if early_stopping_rounds is None else early_stopping_rounds
    )
    resolved_early_stopping_validation_fraction = float(
        0.15
        if early_stopping_validation_fraction is None
        else early_stopping_validation_fraction
    )
    resolved_calibration_method = normalize_probability_calibration_method(calibration_method)
    resolved_initial_threshold = float(
        DEFAULT_INITIAL_THRESHOLD if initial_threshold is None else initial_threshold
    )

    if resolved_bayes_n_iter <= 0:
        raise ValueError("bayes_n_iter 必须为正整数。")
    if resolved_cv_folds <= 1:
        raise ValueError("cv_folds 至少为 2。")
    if not 0.0 < resolved_test_size < 1.0:
        raise ValueError("test_size 必须在 0 和 1 之间。")
    if resolved_search_n_jobs <= 0:
        raise ValueError("search_n_jobs 必须为正整数。")
    if resolved_smote_k_neighbors <= 0:
        raise ValueError("smote_k_neighbors 必须为正整数。")
    if not 0.0 < resolved_smote_sampling_strategy <= 1.0:
        raise ValueError("smote_sampling_strategy 必须在 0 和 1 之间。")
    if resolved_early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds 不能为负数。")
    if not 0.0 < resolved_early_stopping_validation_fraction < 0.5:
        raise ValueError("early_stopping_validation_fraction 必须在 0 和 0.5 之间。")
    if not 0.0 < resolved_initial_threshold < 1.0:
        raise ValueError("initial_threshold 必须在 0 和 1 之间。")

    return NotebookRunConfig(
        bayes_n_iter=resolved_bayes_n_iter,
        cv_folds=resolved_cv_folds,
        random_seed=resolved_random_seed,
        test_size=resolved_test_size,
        device_type=resolved_device_type,
        model_n_jobs=resolved_model_n_jobs,
        search_n_jobs=resolved_search_n_jobs,
        smote_k_neighbors=resolved_smote_k_neighbors,
        smote_sampling_strategy=resolved_smote_sampling_strategy,
        early_stopping_rounds=resolved_early_stopping_rounds,
        early_stopping_validation_fraction=resolved_early_stopping_validation_fraction,
        calibration_method=resolved_calibration_method,
        initial_threshold=resolved_initial_threshold,
    )


def load_training_dataframe(
    data_path: Path,
    random_seed: int,
) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=random_seed)

    return data.reset_index(drop=True)


def _extract_numeric_tokens(value: Any) -> list[float]:
    if pd.isna(value):
        return []
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return [float(value)]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "missing"}:
        return []

    return [float(token) for token in RETENTION_TIME_NUMBER_PATTERN.findall(text)]


def clean_retention_time_series(
    series: pd.Series,
    aggregation: str = "mean",
) -> tuple[pd.Series, Dict[str, Any]]:
    if aggregation != "mean":
        raise ValueError("当前只支持 aggregation='mean'。")

    strict_numeric = pd.to_numeric(series, errors="coerce")
    cleaned_values = []
    multi_value_examples = []
    multi_value_count = 0
    original_missing_count = 0

    for value in series:
        tokens = _extract_numeric_tokens(value)
        if not tokens:
            cleaned_values.append(np.nan)
            if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null", "missing"}:
                original_missing_count += 1
            continue

        if len(tokens) > 1:
            multi_value_count += 1
            if str(value) not in multi_value_examples and len(multi_value_examples) < 5:
                multi_value_examples.append(str(value))

        cleaned_values.append(float(np.mean(tokens)))

    cleaned = pd.Series(cleaned_values, index=series.index, dtype="float64", name=series.name)
    cleaned_missing_count = int(cleaned.isna().sum())
    strict_missing_count = int(pd.isna(strict_numeric).sum())

    diagnostics = {
        "row_count": int(len(series)),
        "strict_missing_count": strict_missing_count,
        "cleaned_missing_count": cleaned_missing_count,
        "recovered_from_text_count": int(max(strict_missing_count - cleaned_missing_count, 0)),
        "original_missing_count": int(original_missing_count),
        "multi_value_count": int(multi_value_count),
        "multi_value_examples": multi_value_examples,
        "aggregation": aggregation,
    }
    diagnostics["unparsed_non_missing_count"] = int(
        max(cleaned_missing_count - diagnostics["original_missing_count"], 0)
    )

    return cleaned, diagnostics


def _replace_dataframe_column(
    frame: pd.DataFrame,
    column: str,
    values: pd.Series | np.ndarray | list[Any],
) -> pd.DataFrame:
    updated = frame.copy()
    updated[column] = values
    return updated


def _normalize_categorical_series(
    series: pd.Series,
    fill_value: Any,
) -> pd.Series:
    return series.fillna(fill_value).astype("string")


def _fit_categorical_encoders(
    X_df: pd.DataFrame,
    categorical_features: list[str],
    categorical_fill_values: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    categorical_encoders: Dict[str, Dict[str, int]] = {}
    categorical_unknown_values: Dict[str, int] = {}

    for column in categorical_features:
        normalized_values = _normalize_categorical_series(
            X_df[column],
            categorical_fill_values[column],
        )
        categories = sorted(normalized_values.dropna().unique().tolist())
        categorical_encoders[column] = {
            category: category_code
            for category_code, category in enumerate(categories)
        }
        categorical_unknown_values[column] = -1

    return categorical_encoders, categorical_unknown_values


def _encode_categorical_features(
    X_df: pd.DataFrame,
    categorical_features: list[str],
    categorical_fill_values: Dict[str, Any],
    categorical_encoders: Dict[str, Dict[str, int]],
    categorical_unknown_values: Dict[str, int],
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=X_df.index)

    for column in categorical_features:
        if column not in categorical_encoders:
            raise ValueError(f"预处理器缺少离散列 {column} 的编码映射，请重新训练模型。")

        normalized_values = _normalize_categorical_series(
            X_df[column],
            categorical_fill_values[column],
        )
        unknown_value = int(categorical_unknown_values.get(column, -1))
        encoded[column] = (
            normalized_values.map(categorical_encoders[column]).fillna(unknown_value).astype("int32")
        )

    return encoded


def _fit_lightgbm_preprocessor(X: pd.DataFrame) -> Dict[str, Any]:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    retention_time_diagnostics = None
    if RETENTION_TIME_COLUMN in X_df.columns:
        cleaned_retention_time, retention_time_diagnostics = clean_retention_time_series(
            X_df[RETENTION_TIME_COLUMN]
        )
        X_df = _replace_dataframe_column(
            X_df,
            RETENTION_TIME_COLUMN,
            cleaned_retention_time.astype("float64"),
        )

    raw_feature_columns = X_df.columns.tolist()
    numeric_features = X_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_df.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_fill_values = {}
    for column in numeric_features:
        mode = X_df[column].mode(dropna=True)
        numeric_fill_values[column] = mode.iloc[0] if not mode.empty else 0

    categorical_fill_values = {}
    for column in categorical_features:
        mode = X_df[column].mode(dropna=True)
        categorical_fill_values[column] = mode.iloc[0] if not mode.empty else "missing"

    continuous_features = numeric_features.copy()
    discrete_features = categorical_features.copy()
    categorical_encoders, categorical_unknown_values = _fit_categorical_encoders(
        X_df,
        discrete_features,
        categorical_fill_values,
    )

    scaler = None
    if continuous_features:
        scaler = StandardScaler()
        filled_continuous = X_df[continuous_features].copy()
        for column in continuous_features:
            filled_continuous.loc[:, column] = pd.to_numeric(
                filled_continuous[column],
                errors="coerce",
            ).astype("float64")
        filled_continuous = filled_continuous.fillna(numeric_fill_values)
        scaler.fit(filled_continuous)

    return {
        "raw_feature_columns": raw_feature_columns,
        "model_feature_columns": continuous_features + discrete_features,
        "continuous_features": continuous_features,
        "discrete_features": discrete_features,
        "categorical_feature_columns": discrete_features.copy(),
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "categorical_encoders": categorical_encoders,
        "categorical_unknown_values": categorical_unknown_values,
        "scaler": scaler,
        "retention_time_column": RETENTION_TIME_COLUMN if RETENTION_TIME_COLUMN in X_df.columns else None,
        "retention_time_multi_value_strategy": "mean",
        "retention_time_diagnostics": retention_time_diagnostics,
    }


def transform_lightgbm_features(
    X: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> pd.DataFrame:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    raw_feature_columns = preprocessor["raw_feature_columns"]
    missing_columns = [column for column in raw_feature_columns if column not in X_df.columns]
    if missing_columns:
        raise ValueError(f"输入数据缺少训练时使用的特征列: {missing_columns}")

    X_aligned = X_df.loc[:, raw_feature_columns].copy()

    retention_time_column = preprocessor.get("retention_time_column")
    if retention_time_column and retention_time_column in X_aligned.columns:
        cleaned_retention_time, _ = clean_retention_time_series(X_aligned[retention_time_column])
        X_aligned = _replace_dataframe_column(
            X_aligned,
            retention_time_column,
            cleaned_retention_time.astype("float64"),
        )

    continuous_features = preprocessor["continuous_features"]
    discrete_features = preprocessor["discrete_features"]
    categorical_fill_values = preprocessor.get("categorical_fill_values", {})
    categorical_encoders = preprocessor.get("categorical_encoders", {})
    categorical_unknown_values = preprocessor.get("categorical_unknown_values", {})

    if continuous_features:
        for column in continuous_features:
            X_aligned.loc[:, column] = pd.to_numeric(
                X_aligned[column],
                errors="coerce",
            ).astype("float64")
        X_aligned.loc[:, continuous_features] = X_aligned[continuous_features].fillna(
            preprocessor["numeric_fill_values"]
        )

    processed = pd.DataFrame(index=X_aligned.index)
    if continuous_features:
        scaler = preprocessor["scaler"]
        scaled_values = scaler.transform(X_aligned[continuous_features])
        processed = pd.DataFrame(
            scaled_values,
            columns=continuous_features,
            index=X_aligned.index,
        )

    if discrete_features:
        encoded_discrete = _encode_categorical_features(
            X_aligned,
            discrete_features,
            categorical_fill_values,
            categorical_encoders,
            categorical_unknown_values,
        )
        processed = pd.concat([processed, encoded_discrete], axis=1)

    processed = processed.loc[:, preprocessor["model_feature_columns"]]
    if processed.isna().any().any():
        raise ValueError("预处理后仍存在缺失值，无法继续执行 SMOTE 和模型训练。")

    return processed.reset_index(drop=True)


def summarize_binary_class_counts(
    y: pd.Series | np.ndarray | list[Any],
) -> Dict[int, int]:
    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y, name="target")
    counts = y_series.value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def compute_positive_class_ratio(
    y: pd.Series | np.ndarray | list[Any],
) -> float:
    class_counts = summarize_binary_class_counts(y)
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))
    if negative_count <= 0:
        return 1.0
    return float(positive_count / negative_count)


def compute_scale_pos_weight_from_labels(
    y: pd.Series | np.ndarray | list[Any],
) -> float:
    class_counts = summarize_binary_class_counts(y)
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))
    if positive_count <= 0:
        raise ValueError("训练数据中缺少正类样本，无法自动计算 scale_pos_weight。")
    return float(negative_count / positive_count)


def build_booster_feature_importance_frame(
    booster: Any,
    importance_type: str = "split",
    ignore_zero: bool = False,
) -> pd.DataFrame:
    if booster is None:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = list(booster.feature_name())
    importance_values = np.asarray(
        booster.feature_importance(importance_type=importance_type),
        dtype="float64",
    )
    if not feature_names or importance_values.size == 0:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(feature_names) != int(importance_values.size):
        raise ValueError("LightGBM booster 返回的特征名数量与 importance 数量不一致。")

    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )
    if ignore_zero:
        importance_frame = importance_frame.loc[importance_frame["importance"] > 0]
    if importance_frame.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    return importance_frame.sort_values(
        by=["importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def summarize_lightgbm_booster_training_diagnostics(
    booster: Any,
) -> Dict[str, Any]:
    split_importance = build_booster_feature_importance_frame(
        booster,
        importance_type="split",
        ignore_zero=False,
    )
    gain_importance = build_booster_feature_importance_frame(
        booster,
        importance_type="gain",
        ignore_zero=False,
    )

    if split_importance.empty:
        total_split_count = 0
        non_zero_split_feature_count = 0
    else:
        split_values = split_importance["importance"].astype("float64")
        total_split_count = int(split_values.sum())
        non_zero_split_feature_count = int((split_values > 0).sum())

    total_gain = (
        float(gain_importance["importance"].astype("float64").sum())
        if not gain_importance.empty
        else 0.0
    )
    tree_count = int(booster.num_trees()) if booster is not None else 0

    return {
        "tree_count": tree_count,
        "total_split_count": total_split_count,
        "non_zero_split_feature_count": non_zero_split_feature_count,
        "total_gain": total_gain,
        "is_degenerate": bool(total_split_count <= 0),
    }


def resample_training_fold_with_borderline_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
    sampling_strategy: float = DEFAULT_SMOTE_SAMPLING_STRATEGY,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y, name="target")
    y_series = y_series.reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    if X_df.select_dtypes(exclude=[np.number]).shape[1] > 0:
        raise ValueError("当前 Borderline-SMOTE 路径要求输入特征全部为数值列。")

    class_counts = summarize_binary_class_counts(y_series)
    minority_count = int(min(class_counts.values(), default=0))
    majority_count = int(max(class_counts.values(), default=0))
    minority_ratio = float(minority_count / majority_count) if majority_count > 0 else 1.0
    diagnostics = {
        "resampler": "BorderlineSMOTE",
        "resampler_kind": DEFAULT_BORDERLINE_SMOTE_KIND,
        "sampling_strategy_target": float(sampling_strategy),
        "input_class_counts": class_counts,
        "output_class_counts": class_counts,
        "positive_ratio_before": compute_positive_class_ratio(y_series),
        "positive_ratio_after": compute_positive_class_ratio(y_series),
        "applied": False,
        "skip_reason": None,
    }

    if len(class_counts) < 2:
        diagnostics["skip_reason"] = "单类训练折无法执行重采样。"
        return X_df, y_series, diagnostics

    if minority_count <= 1:
        diagnostics["skip_reason"] = "少数类样本不足 2 条，无法执行 Borderline-SMOTE。"
        return X_df, y_series, diagnostics

    if minority_ratio >= float(sampling_strategy):
        diagnostics["skip_reason"] = "当前训练折的少数类占比已不低于目标采样比例。"
        return X_df, y_series, diagnostics

    safe_k_neighbors = min(int(k_neighbors), minority_count - 1)
    if safe_k_neighbors <= 0:
        diagnostics["skip_reason"] = "可用少数类近邻不足，无法执行 Borderline-SMOTE。"
        return X_df, y_series, diagnostics

    max_total_neighbors = max(1, len(y_series) - 1)
    safe_m_neighbors = min(max_total_neighbors, max(2, safe_k_neighbors + 1))

    try:
        from imblearn.over_sampling import BorderlineSMOTE
    except ImportError as exc:  # pragma: no cover - 依赖是否存在取决于运行环境
        raise ImportError(
            "当前训练路径需要 imbalanced-learn 才能执行 Borderline-SMOTE，请先安装 requirements.txt 中的新依赖。"
        ) from exc

    try:
        sampler = BorderlineSMOTE(
            sampling_strategy=float(sampling_strategy),
            random_state=random_state,
            k_neighbors=safe_k_neighbors,
            m_neighbors=safe_m_neighbors,
            kind=DEFAULT_BORDERLINE_SMOTE_KIND,
        )
        X_resampled, y_resampled = sampler.fit_resample(X_df, y_series)
    except ValueError as exc:
        diagnostics["skip_reason"] = f"Borderline-SMOTE 未成功执行: {exc}"
        return X_df, y_series, diagnostics

    X_resampled_df = pd.DataFrame(X_resampled, columns=X_df.columns).reset_index(drop=True)
    y_resampled_series = pd.Series(y_resampled, name=y_series.name).reset_index(drop=True)

    diagnostics["output_class_counts"] = summarize_binary_class_counts(y_resampled_series)
    diagnostics["positive_ratio_after"] = compute_positive_class_ratio(y_resampled_series)
    diagnostics["applied"] = True

    return X_resampled_df, y_resampled_series, diagnostics


def compute_binary_classification_metrics(
    y_true: pd.Series | np.ndarray | list[Any],
    positive_proba: pd.Series | np.ndarray | list[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    positive_proba_array = np.asarray(positive_proba, dtype=float)
    resolved_threshold = float(threshold)
    y_pred_array = (positive_proba_array >= resolved_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
    ).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics = {
        "threshold": resolved_threshold,
        "AUC": float(roc_auc_score(y_true_array, positive_proba_array)),
        "Accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "Precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "Recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "F1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "Specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=int),
    }
    return metrics


def probe_binary_classification_thresholds(
    y_true: pd.Series | np.ndarray | list[Any],
    positive_proba: pd.Series | np.ndarray | list[float],
    thresholds: Optional[list[float] | np.ndarray] = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.linspace(0.30, 0.70, 41), 3)

    rows = []
    for threshold in thresholds:
        metrics = compute_binary_classification_metrics(
            y_true=y_true,
            positive_proba=positive_proba,
            threshold=float(threshold),
        )
        rows.append(
            {
                key: value
                for key, value in metrics.items()
                if key not in {"confusion_matrix"}
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def compute_probability_calibration_metrics(
    y_true: pd.Series | np.ndarray | list[Any],
    positive_proba: pd.Series | np.ndarray | list[float],
) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    positive_proba_array = np.clip(np.asarray(positive_proba, dtype=float), 0.0, 1.0)
    metrics = {
        "brier_score": float(brier_score_loss(y_true_array, positive_proba_array)),
    }
    if np.unique(y_true_array).size >= 2:
        metrics["auc"] = float(roc_auc_score(y_true_array, positive_proba_array))
    else:
        metrics["auc"] = float("nan")
    return metrics


def apply_probability_calibrator(
    positive_proba: pd.Series | np.ndarray | list[float],
    calibration_bundle: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    positive_proba_array = np.clip(np.asarray(positive_proba, dtype=float), 0.0, 1.0)
    if not calibration_bundle:
        return positive_proba_array

    calibrator = calibration_bundle.get("calibrator")
    resolved_method = normalize_probability_calibration_method(
        calibration_bundle.get("method", calibration_bundle.get("requested_method"))
    )
    if calibrator is None:
        return positive_proba_array

    if resolved_method == "isotonic":
        calibrated = calibrator.predict(positive_proba_array)
    else:
        calibrated = calibrator.predict_proba(positive_proba_array.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def fit_probability_calibrator(
    y_true: pd.Series | np.ndarray | list[Any],
    positive_proba: pd.Series | np.ndarray | list[float],
    method: str = DEFAULT_CALIBRATION_METHOD,
    random_state: int = 42,
) -> Dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    positive_proba_array = np.clip(np.asarray(positive_proba, dtype=float), 0.0, 1.0)
    if np.unique(y_true_array).size < 2:
        raise ValueError("概率校准要求输入标签同时包含正类与负类。")

    requested_method = normalize_probability_calibration_method(method)
    class_counts = summarize_binary_class_counts(y_true_array)
    min_class_count = int(min(class_counts.values()))
    unique_score_count = int(np.unique(np.round(positive_proba_array, 12)).size)

    fallback_reason = None
    resolved_method = requested_method
    if requested_method == "auto":
        resolved_method = (
            "isotonic" if min_class_count >= 30 and unique_score_count >= 20 else "platt"
        )
    elif requested_method == "isotonic" and (min_class_count < 10 or unique_score_count < 10):
        resolved_method = "platt"
        fallback_reason = "OOF 分数离散度或最小类别样本数不足，自动退回到 Platt Scaling。"

    if resolved_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(positive_proba_array, y_true_array)
    else:
        calibrator = LogisticRegression(
            random_state=random_state,
            solver="lbfgs",
            max_iter=1000,
        )
        calibrator.fit(positive_proba_array.reshape(-1, 1), y_true_array)

    calibration_bundle = {
        "requested_method": requested_method,
        "method": resolved_method,
        "fallback_reason": fallback_reason,
        "calibrator": calibrator,
        "sample_count": int(len(y_true_array)),
        "class_counts": class_counts,
        "unique_score_count": unique_score_count,
        "raw_metrics": compute_probability_calibration_metrics(y_true_array, positive_proba_array),
    }
    calibration_bundle["calibrated_positive_proba"] = apply_probability_calibrator(
        positive_proba_array,
        calibration_bundle,
    )
    calibration_bundle["calibrated_metrics"] = compute_probability_calibration_metrics(
        y_true_array,
        calibration_bundle["calibrated_positive_proba"],
    )
    return calibration_bundle


def select_binary_classification_threshold(
    threshold_frame: pd.DataFrame,
    primary_metric: str = "F1",
    initial_threshold: float = DEFAULT_INITIAL_THRESHOLD,
) -> Dict[str, Any]:
    allowed_metrics = {
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
    }
    if primary_metric not in allowed_metrics:
        raise ValueError(f"threshold 选择指标必须属于 {sorted(allowed_metrics)}")

    ranked = threshold_frame.copy()
    ranked["distance_to_initial_threshold"] = (
        ranked["threshold"].astype(float) - float(initial_threshold)
    ).abs()
    priority_map = {
        "F1": ["F1", "Balanced Accuracy", "Recall", "Specificity"],
        "Balanced Accuracy": ["Balanced Accuracy", "F1", "Recall", "Specificity"],
        "Recall": ["Recall", "F1", "Balanced Accuracy", "Specificity"],
        "Precision": ["Precision", "F1", "Balanced Accuracy", "Recall"],
        "Specificity": ["Specificity", "Balanced Accuracy", "F1", "Recall"],
        "Accuracy": ["Accuracy", "Balanced Accuracy", "F1", "Recall"],
    }
    sort_by = priority_map[primary_metric] + ["distance_to_initial_threshold", "threshold"]
    ascending = [False] * len(priority_map[primary_metric]) + [True, True]
    selected = (
        ranked.sort_values(by=sort_by, ascending=ascending)
        .iloc[0]
        .drop(labels=["distance_to_initial_threshold"])
        .to_dict()
    )
    selected["selection_metric"] = primary_metric
    selected["initial_threshold"] = float(initial_threshold)
    return selected


# sklearn>=1.6 通过 tags 判断 estimator 类型，mixin 需要放在 BaseEstimator 左侧。
class FoldSafeSmoteLGBMClassifier(ClassifierMixin, BaseEstimator):
    """
    LightGBM 二分类封装器。

    说明：
    - 保留历史类名以兼容旧 notebook，但内部实现已经收敛成单一路径。
    - 每个 `fit` 都会在当前训练折内部重新学习填充值与标准化器，避免任何 CV 泄漏。
    - 先在训练折内部再切一小块 early stopping 验证集；这块验证集绝不参与重采样。
    - 仅对当前折的训练子集执行 Borderline-SMOTE 增样，用来扩充训练样本量。
    - `scale_pos_weight` 强制按“重采样后的新标签分布”自动计算，不再允许使用原始比例或手工覆盖。
    - 外层验证集、OOF 与最终测试集都保持原始不平衡分布，用于真实评估。
    """

    def __init__(
        self,
        random_state: int = 42,
        device_type: str = DEFAULT_LGBM_DEVICE_TYPE,
        model_n_jobs: int = 8,
        smote_k_neighbors: int = 5,
        smote_sampling_strategy: float = DEFAULT_SMOTE_SAMPLING_STRATEGY,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 300,
        early_stopping_validation_fraction: float = 0.15,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        max_depth: int = -1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_samples: int = 20,
        min_split_gain: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
    ):
        self.random_state = random_state
        self.device_type = normalize_lightgbm_device_type(device_type)
        self.model_n_jobs = model_n_jobs
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_sampling_strategy = smote_sampling_strategy
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_validation_fraction = early_stopping_validation_fraction
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.min_split_gain = min_split_gain
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def _resolve_training_scale_pos_weight(self, y_train: pd.Series) -> float:
        if self.scale_pos_weight is not None:
            raise ValueError(
                "当前训练路径强制根据折内重采样后的标签分布自动计算 scale_pos_weight，不支持手动指定。"
            )
        return compute_scale_pos_weight_from_labels(y_train)

    def _split_early_stopping_validation(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.early_stopping_rounds <= 0:
            return X_df, pd.DataFrame(), y_series, pd.Series(dtype=y_series.dtype)

        if not 0.0 < float(self.early_stopping_validation_fraction) < 0.5:
            raise ValueError("early_stopping_validation_fraction 必须在 0 和 0.5 之间。")

        class_counts = y_series.value_counts()
        if len(y_series) < 10 or class_counts.min() < 2:
            return X_df, pd.DataFrame(), y_series, pd.Series(dtype=y_series.dtype)

        try:
            X_fit, X_eval, y_fit, y_eval = train_test_split(
                X_df,
                y_series,
                test_size=float(self.early_stopping_validation_fraction),
                random_state=self.random_state,
                stratify=y_series,
            )
        except ValueError:
            return X_df, pd.DataFrame(), y_series, pd.Series(dtype=y_series.dtype)
        return (
            X_fit.reset_index(drop=True),
            X_eval.reset_index(drop=True),
            y_fit.reset_index(drop=True),
            y_eval.reset_index(drop=True),
        )

    def _build_model(self, effective_scale_pos_weight: float = 1.0):
        resolved_device_type = normalize_lightgbm_device_type(self.device_type)
        self.model_device_type_ = resolved_device_type
        self.model_max_bin_ = resolve_lightgbm_max_bin(resolved_device_type)

        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            device_type=resolved_device_type,
            random_state=self.random_state,
            n_jobs=self.model_n_jobs,
            subsample_freq=1,
            max_bin=self.model_max_bin_,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            min_split_gain=self.min_split_gain,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=effective_scale_pos_weight,
            verbosity=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.smote_k_neighbors <= 0:
            raise ValueError("smote_k_neighbors 必须为正整数。")
        if not 0.0 < float(self.smote_sampling_strategy) <= 1.0:
            raise ValueError("smote_sampling_strategy 必须在 0 和 1 之间。")
        if self.early_stopping_rounds < 0:
            raise ValueError("early_stopping_rounds 不能为负数。")
        if self.model_n_jobs <= 0:
            raise ValueError("model_n_jobs 必须为正整数。")

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y, name="target")
        y_series = y_series.reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        self.fit_class_counts_ = summarize_binary_class_counts(y_series)

        X_fit, X_eval, y_fit, y_eval = self._split_early_stopping_validation(X_df, y_series)
        self.train_split_class_counts_ = summarize_binary_class_counts(y_fit)
        self.eval_split_class_counts_ = summarize_binary_class_counts(y_eval)

        self.preprocessor_bundle_ = _fit_lightgbm_preprocessor(X_fit)
        X_fit_processed = transform_lightgbm_features(X_fit, self.preprocessor_bundle_)
        X_eval_processed = None
        if not X_eval.empty:
            X_eval_processed = transform_lightgbm_features(X_eval, self.preprocessor_bundle_)

        X_model_fit, y_model_fit, self.resampling_metadata_ = resample_training_fold_with_borderline_smote(
            X_fit_processed,
            y_fit,
            random_state=self.random_state,
            k_neighbors=self.smote_k_neighbors,
            sampling_strategy=self.smote_sampling_strategy,
        )

        self.pre_resample_scale_pos_weight_ = compute_scale_pos_weight_from_labels(y_fit)
        self.effective_scale_pos_weight_ = self._resolve_training_scale_pos_weight(y_model_fit)
        self.scale_pos_weight_source_ = "post_resample_ratio"
        self.model_fit_class_counts_ = summarize_binary_class_counts(y_model_fit)
        self.resampled_class_counts_ = self.model_fit_class_counts_
        self.resampler_name_ = self.resampling_metadata_.get("resampler")
        self.training_positive_ratio_before_resampling_ = self.resampling_metadata_.get(
            "positive_ratio_before"
        )
        self.training_positive_ratio_after_resampling_ = self.resampling_metadata_.get(
            "positive_ratio_after"
        )
        self.used_early_stopping_ = bool(self.early_stopping_rounds > 0 and X_eval_processed is not None)

        self.model_ = self._build_model(
            effective_scale_pos_weight=self.effective_scale_pos_weight_,
        )
        fit_kwargs: Dict[str, Any] = {}
        if self.used_early_stopping_:
            import lightgbm as lgb

            fit_kwargs = {
                "eval_set": [(X_eval_processed, y_eval)],
                "eval_names": ["fold_valid"],
                "eval_metric": "auc",
                "callbacks": [
                    lgb.early_stopping(
                        stopping_rounds=int(self.early_stopping_rounds),
                        verbose=False,
                    )
                ],
            }
        categorical_feature_columns = self.preprocessor_bundle_.get("categorical_feature_columns", [])
        if categorical_feature_columns:
            fit_kwargs["categorical_feature"] = categorical_feature_columns

        self.model_.fit(X_model_fit, y_model_fit, **fit_kwargs)
        self.classes_ = getattr(self.model_, "classes_", np.sort(y_series.unique()))
        self.booster_ = self.model_.booster_
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        self.training_diagnostics_ = summarize_lightgbm_booster_training_diagnostics(
            self.booster_
        )
        self.total_split_count_ = int(self.training_diagnostics_["total_split_count"])
        self.non_zero_split_feature_count_ = int(
            self.training_diagnostics_["non_zero_split_feature_count"]
        )
        self.total_gain_ = float(self.training_diagnostics_["total_gain"])
        if self.training_diagnostics_["is_degenerate"]:
            raise RuntimeError(
                "当前超参数组合训练出的 LightGBM 未产生任何有效分裂，模型已崩塌。"
                f" best_iteration={self.best_iteration_},"
                f" total_split_count={self.total_split_count_},"
                f" device_type={self.model_device_type_},"
                f" max_bin={self.model_max_bin_},"
                f" fit_class_counts={self.fit_class_counts_},"
                f" resampled_class_counts={self.model_fit_class_counts_},"
                f" scale_pos_weight={self.effective_scale_pos_weight_:.6f},"
                f" num_leaves={self.num_leaves},"
                f" max_depth={self.max_depth},"
                f" min_child_samples={self.min_child_samples},"
                f" min_split_gain={self.min_split_gain},"
                f" reg_alpha={self.reg_alpha},"
                f" reg_lambda={self.reg_lambda}."
            )

        return self

    def predict(self, X: pd.DataFrame):
        check_is_fitted(self, "model_")
        X_processed = transform_lightgbm_features(X, self.preprocessor_bundle_)
        return self.model_.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame):
        check_is_fitted(self, "model_")
        X_processed = transform_lightgbm_features(X, self.preprocessor_bundle_)
        return self.model_.predict_proba(X_processed)

    def get_preprocessor_bundle(self) -> Dict[str, Any]:
        check_is_fitted(self, "preprocessor_bundle_")
        return self.preprocessor_bundle_


def prepare_lightgbm_training_data(
    data: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    random_state: int = 42,
    test_size: float = 0.2,
    smote_k_neighbors: int = 5,
) -> Dict[str, Any]:
    if smote_k_neighbors <= 0:
        raise ValueError("smote_k_neighbors 必须为正整数。")

    frame = data.copy()
    retention_time_diagnostics = None
    if RETENTION_TIME_COLUMN in frame.columns:
        cleaned_retention_time, retention_time_diagnostics = clean_retention_time_series(
            frame[RETENTION_TIME_COLUMN]
        )
        frame = _replace_dataframe_column(
            frame,
            RETENTION_TIME_COLUMN,
            cleaned_retention_time.astype("float64"),
        )

    X = frame.drop([target_column], axis=1)
    y = frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "raw_feature_columns": X.columns.tolist(),
        "retention_time_diagnostics": retention_time_diagnostics,
        "smote_k_neighbors": smote_k_neighbors,
    }


def validate_cuda_only_requested(device_type: str = "cuda") -> None:
    resolved_device_type = normalize_lightgbm_device_type(device_type)
    if resolved_device_type != "cuda":
        raise RuntimeError(
            "当前 helper 仅用于 CUDA 专用预检；如需 GPU/OpenCL 预检，请改用 "
            "validate_lightgbm_accelerated_build(device_type='gpu')."
        )


def validate_accelerated_device_requested(device_type: Optional[str] = None) -> str:
    return normalize_lightgbm_device_type(device_type)


def validate_lightgbm_accelerated_build(
    device_type: Optional[str] = None,
    random_state: int = 42,
) -> str:
    resolved_device_type = validate_accelerated_device_requested(device_type)

    import lightgbm as lgb

    X_probe = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    y_probe = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)

    probe = lgb.LGBMClassifier(
        objective="binary",
        device_type=resolved_device_type,
        n_estimators=4,
        num_leaves=7,
        max_depth=3,
        max_bin=resolve_lightgbm_max_bin(resolved_device_type),
        min_data_in_leaf=1,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )

    try:
        probe.fit(X_probe, y_probe)
        probe.predict_proba(X_probe)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        lgbm_version = getattr(lgb, "__version__", "unknown")
        build_requirement = "启用 USE_CUDA=1 的构建" if resolved_device_type == "cuda" else "启用 USE_GPU=ON / OpenCL 的构建"
        raise RuntimeError(
            f"当前 LightGBM 不能以 {resolved_device_type} 模式训练。\n"
            f"已检测到 lightgbm=={lgbm_version}，但它不是 {build_requirement}。\n"
            "项目已禁用任何静默 fallback。\n"
            f"{get_lightgbm_device_installation_notes(resolved_device_type)}"
        ) from exc

    return getattr(lgb, "__version__", "unknown")


def validate_lightgbm_cuda_build(
    device_type: str = "cuda",
    random_state: int = 42,
) -> str:
    return validate_lightgbm_accelerated_build(
        device_type=device_type,
        random_state=random_state,
    )


def build_lgbm_classifier(
    random_state: int = 42,
    device_type: str = DEFAULT_LGBM_DEVICE_TYPE,
    model_n_jobs: Optional[int] = None,
    smote_k_neighbors: int = 5,
    smote_sampling_strategy: float = DEFAULT_SMOTE_SAMPLING_STRATEGY,
    scale_pos_weight: Optional[float] = None,
    early_stopping_rounds: int = 300,
    early_stopping_validation_fraction: float = 0.15,
):
    return FoldSafeSmoteLGBMClassifier(
        random_state=random_state,
        device_type=device_type,
        model_n_jobs=resolve_recommended_model_n_jobs(model_n_jobs),
        smote_k_neighbors=smote_k_neighbors,
        smote_sampling_strategy=smote_sampling_strategy,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        early_stopping_validation_fraction=early_stopping_validation_fraction,
    )


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compact_probability_calibration_bundle(
    probability_calibration_bundle: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not probability_calibration_bundle:
        return None

    compact_bundle = dict(probability_calibration_bundle)
    compact_bundle.pop("calibrated_positive_proba", None)
    return compact_bundle


def _probability_calibration_manifest_metadata(
    probability_calibration_bundle: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    compact_bundle = _compact_probability_calibration_bundle(probability_calibration_bundle)
    if not compact_bundle:
        return None

    manifest_bundle = dict(compact_bundle)
    manifest_bundle.pop("calibrator", None)
    return _make_json_safe(manifest_bundle)


def save_lightgbm_inference_artifacts(
    estimator: Any,
    prepared: Dict[str, Any],
    model_path: Path,
    preprocessor_path: Path,
    manifest_path: Path,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    data_path: Optional[Path] = None,
    random_seed: Optional[int] = None,
    test_size: Optional[float] = None,
    smote_k_neighbors: Optional[int] = None,
    smote_sampling_strategy: Optional[float] = None,
    scoring: Optional[str] = None,
    classification_threshold: float = DEFAULT_INITIAL_THRESHOLD,
    probability_calibration_bundle: Optional[Dict[str, Any]] = None,
    threshold_selection_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    manifest_path = Path(manifest_path)

    for path in (model_path, preprocessor_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    estimator.booster_.save_model(str(model_path))

    estimator_preprocessor = (
        estimator.get_preprocessor_bundle()
        if hasattr(estimator, "get_preprocessor_bundle")
        else {}
    )
    compact_probability_calibration_bundle = _compact_probability_calibration_bundle(
        probability_calibration_bundle
    )
    probability_calibration_metadata = _probability_calibration_manifest_metadata(
        probability_calibration_bundle
    )
    preprocessor_bundle = {
        "artifacts_version": ARTIFACTS_VERSION,
        "target_column": target_column,
        "raw_feature_columns": estimator_preprocessor.get(
            "raw_feature_columns",
            prepared.get("raw_feature_columns", []),
        ),
        "model_feature_columns": estimator_preprocessor.get(
            "model_feature_columns",
            prepared.get("model_feature_columns", []),
        ),
        "continuous_features": estimator_preprocessor.get(
            "continuous_features",
            prepared.get("continuous_features", []),
        ),
        "discrete_features": estimator_preprocessor.get(
            "discrete_features",
            prepared.get("discrete_features", []),
        ),
        "categorical_feature_columns": estimator_preprocessor.get(
            "categorical_feature_columns",
            prepared.get("categorical_feature_columns", []),
        ),
        "numeric_fill_values": estimator_preprocessor.get(
            "numeric_fill_values",
            prepared.get("numeric_fill_values", {}),
        ),
        "categorical_fill_values": estimator_preprocessor.get(
            "categorical_fill_values",
            prepared.get("categorical_fill_values", {}),
        ),
        "categorical_encoders": estimator_preprocessor.get(
            "categorical_encoders",
            prepared.get("categorical_encoders", {}),
        ),
        "categorical_unknown_values": estimator_preprocessor.get(
            "categorical_unknown_values",
            prepared.get("categorical_unknown_values", {}),
        ),
        "scaler": estimator_preprocessor.get("scaler", prepared.get("scaler")),
        "retention_time_column": estimator_preprocessor.get("retention_time_column"),
        "retention_time_multi_value_strategy": estimator_preprocessor.get(
            "retention_time_multi_value_strategy"
        ),
        "retention_time_diagnostics": estimator_preprocessor.get("retention_time_diagnostics"),
        "classes_": list(getattr(estimator, "classes_", [])),
        "classification_threshold": classification_threshold,
        "threshold_selection_metadata": threshold_selection_metadata,
        "probability_calibration_bundle": compact_probability_calibration_bundle,
        "device_type": getattr(estimator, "model_device_type_", getattr(estimator, "device_type", None)),
        "max_bin": getattr(estimator, "model_max_bin_", None),
        "resampler_name": getattr(estimator, "resampler_name_", None),
        "resampling_metadata": getattr(estimator, "resampling_metadata_", None),
        "training_diagnostics": getattr(estimator, "training_diagnostics_", None),
        "pre_resample_scale_pos_weight": getattr(estimator, "pre_resample_scale_pos_weight_", None),
        "effective_scale_pos_weight": getattr(estimator, "effective_scale_pos_weight_", None),
        "scale_pos_weight_source": getattr(estimator, "scale_pos_weight_source_", None),
        "early_stopping_rounds": getattr(estimator, "early_stopping_rounds", None),
        "early_stopping_validation_fraction": getattr(
            estimator,
            "early_stopping_validation_fraction",
            None,
        ),
        "best_iteration": getattr(estimator, "best_iteration_", None),
    }
    joblib.dump(preprocessor_bundle, preprocessor_path)

    manifest = {
        "artifacts_version": ARTIFACTS_VERSION,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "target_column": target_column,
        "data_path": data_path,
        "raw_feature_columns": preprocessor_bundle["raw_feature_columns"],
        "model_feature_columns": preprocessor_bundle["model_feature_columns"],
        "continuous_features": preprocessor_bundle["continuous_features"],
        "discrete_features": preprocessor_bundle["discrete_features"],
        "categorical_feature_columns": preprocessor_bundle["categorical_feature_columns"],
        "numeric_fill_values": preprocessor_bundle["numeric_fill_values"],
        "categorical_fill_values": preprocessor_bundle["categorical_fill_values"],
        "categorical_encoders": preprocessor_bundle["categorical_encoders"],
        "categorical_unknown_values": preprocessor_bundle["categorical_unknown_values"],
        "retention_time_column": preprocessor_bundle["retention_time_column"],
        "retention_time_multi_value_strategy": preprocessor_bundle[
            "retention_time_multi_value_strategy"
        ],
        "retention_time_diagnostics": preprocessor_bundle["retention_time_diagnostics"],
        "dataset_retention_time_diagnostics": prepared.get("retention_time_diagnostics"),
        "classes_": list(getattr(estimator, "classes_", [])),
        "classification_threshold": classification_threshold,
        "threshold_selection_metadata": threshold_selection_metadata,
        "probability_calibration": probability_calibration_metadata,
        "device_type": getattr(estimator, "model_device_type_", getattr(estimator, "device_type", None)),
        "max_bin": getattr(estimator, "model_max_bin_", None),
        "resampler_name": getattr(estimator, "resampler_name_", None),
        "resampling_metadata": getattr(estimator, "resampling_metadata_", None),
        "training_diagnostics": getattr(estimator, "training_diagnostics_", None),
        "pre_resample_scale_pos_weight": getattr(estimator, "pre_resample_scale_pos_weight_", None),
        "effective_scale_pos_weight": getattr(estimator, "effective_scale_pos_weight_", None),
        "scale_pos_weight_source": getattr(estimator, "scale_pos_weight_source_", None),
        "early_stopping_rounds": getattr(estimator, "early_stopping_rounds", None),
        "early_stopping_validation_fraction": getattr(
            estimator,
            "early_stopping_validation_fraction",
            None,
        ),
        "best_iteration": getattr(estimator, "best_iteration_", None),
        "random_seed": random_seed,
        "test_size": test_size,
        "smote_k_neighbors": smote_k_neighbors,
        "smote_sampling_strategy": smote_sampling_strategy,
        "scoring": scoring,
    }
    manifest_path.write_text(
        json.dumps(_make_json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "manifest_path": manifest_path,
    }
