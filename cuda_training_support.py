from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


DEFAULT_DATA_FILE = "质谱数据汇总_处理后后后2.csv"
DEFAULT_TARGET_COLUMN = "毒性"
RETENTION_TIME_COLUMN = "RETENTION_TIME"
RETENTION_TIME_NUMBER_PATTERN = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")
LIGHTGBM_CUDA_PIP_INSTALL_COMMAND = (
    "pip install lightgbm --no-binary lightgbm "
    "--config-settings=cmake.define.USE_CUDA=ON"
)
LIGHTGBM_CUDA_SOURCE_BUILD_COMMANDS = (
    "git clone --recursive https://github.com/microsoft/LightGBM\n"
    "cd LightGBM\n"
    "cmake -B build -S . -DUSE_CUDA=ON\n"
    "cmake --build build -j4"
)


@dataclass(frozen=True)
class NotebookRunConfig:
    bayes_n_iter: int
    cv_folds: int
    random_seed: int
    test_size: float
    model_n_jobs: int
    search_n_jobs: int
    smote_k_neighbors: int
    early_stopping_rounds: int
    early_stopping_validation_fraction: float


def format_notebook_run_summary(
    config: NotebookRunConfig,
    data_path: Optional[Path] = None,
) -> str:
    lines = [
        f"随机种子: {config.random_seed}",
        f"测试集比例: {config.test_size}",
        f"BayesSearch n_iter: {config.bayes_n_iter}",
        f"CV folds: {config.cv_folds}",
        f"模型 n_jobs: {config.model_n_jobs}",
        f"搜索 n_jobs: {config.search_n_jobs}",
        f"SMOTE k_neighbors: {config.smote_k_neighbors}",
        f"折内 early stopping rounds: {config.early_stopping_rounds}",
        f"折内 early stopping 验证集比例: {config.early_stopping_validation_fraction}",
    ]
    if data_path is not None:
        lines.insert(1, f"数据文件: {data_path}")
    return "\n".join(lines)


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
    model_n_jobs: Optional[int] = None,
    search_n_jobs: Optional[int] = None,
    smote_k_neighbors: Optional[int] = None,
    early_stopping_rounds: Optional[int] = None,
    early_stopping_validation_fraction: Optional[float] = None,
) -> NotebookRunConfig:
    resolved_random_seed = int(42 if random_seed is None else random_seed)
    resolved_test_size = float(0.2 if test_size is None else test_size)
    resolved_bayes_n_iter = int(24 if bayes_n_iter is None else bayes_n_iter)
    resolved_cv_folds = int(5 if cv_folds is None else cv_folds)

    # GPU 训练与交叉验证并发通常会争抢同一张卡，这里默认串行执行搜索。
    resolved_model_n_jobs = int(1 if model_n_jobs is None else model_n_jobs)
    resolved_search_n_jobs = int(1 if search_n_jobs is None else search_n_jobs)
    resolved_smote_k_neighbors = int(5 if smote_k_neighbors is None else smote_k_neighbors)
    resolved_early_stopping_rounds = int(
        100 if early_stopping_rounds is None else early_stopping_rounds
    )
    resolved_early_stopping_validation_fraction = float(
        0.15
        if early_stopping_validation_fraction is None
        else early_stopping_validation_fraction
    )

    if resolved_bayes_n_iter <= 0:
        raise ValueError("bayes_n_iter 必须为正整数。")
    if resolved_cv_folds <= 1:
        raise ValueError("cv_folds 至少为 2。")
    if not 0.0 < resolved_test_size < 1.0:
        raise ValueError("test_size 必须在 0 和 1 之间。")
    if resolved_model_n_jobs <= 0 or resolved_search_n_jobs <= 0:
        raise ValueError("model_n_jobs 和 search_n_jobs 必须为正整数。")
    if resolved_smote_k_neighbors <= 0:
        raise ValueError("smote_k_neighbors 必须为正整数。")
    if resolved_early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds 不能为负数。")
    if not 0.0 < resolved_early_stopping_validation_fraction < 0.5:
        raise ValueError("early_stopping_validation_fraction 必须在 0 和 0.5 之间。")

    return NotebookRunConfig(
        bayes_n_iter=resolved_bayes_n_iter,
        cv_folds=resolved_cv_folds,
        random_seed=resolved_random_seed,
        test_size=resolved_test_size,
        model_n_jobs=resolved_model_n_jobs,
        search_n_jobs=resolved_search_n_jobs,
        smote_k_neighbors=resolved_smote_k_neighbors,
        early_stopping_rounds=resolved_early_stopping_rounds,
        early_stopping_validation_fraction=resolved_early_stopping_validation_fraction,
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
    strict_missing_count = int(strict_numeric.isna().sum())

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


def _fit_lightgbm_preprocessor(X: pd.DataFrame) -> Dict[str, Any]:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    retention_time_diagnostics = None
    if RETENTION_TIME_COLUMN in X_df.columns:
        cleaned_retention_time, retention_time_diagnostics = clean_retention_time_series(
            X_df[RETENTION_TIME_COLUMN]
        )
        X_df.loc[:, RETENTION_TIME_COLUMN] = cleaned_retention_time

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
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
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
        X_aligned.loc[:, retention_time_column] = cleaned_retention_time

    continuous_features = preprocessor["continuous_features"]
    discrete_features = preprocessor["discrete_features"]

    if continuous_features:
        for column in continuous_features:
            X_aligned.loc[:, column] = pd.to_numeric(
                X_aligned[column],
                errors="coerce",
            ).astype("float64")
        X_aligned.loc[:, continuous_features] = X_aligned[continuous_features].fillna(
            preprocessor["numeric_fill_values"]
        )

    if discrete_features:
        X_aligned.loc[:, discrete_features] = X_aligned[discrete_features].fillna(
            preprocessor["categorical_fill_values"]
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
        processed = pd.concat([processed, X_aligned[discrete_features]], axis=1)

    processed = processed.loc[:, preprocessor["model_feature_columns"]]
    if processed.isna().any().any():
        raise ValueError("预处理后仍存在缺失值，无法继续执行 SMOTE 和模型训练。")

    return processed.reset_index(drop=True)


def notebook_smote_resample(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y, name="target")
    y_series = y_series.reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    class_counts = y_series.value_counts()
    target_count = int(class_counts.max())
    rng = np.random.default_rng(random_state)

    numeric_columns = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X_df.columns if col not in numeric_columns]

    generated_batches = []
    generated_labels = []

    for class_label, class_count in class_counts.items():
        n_to_generate = target_count - int(class_count)
        if n_to_generate <= 0:
            continue

        class_rows = X_df.loc[y_series == class_label].reset_index(drop=True)
        if len(class_rows) == 1:
            synthetic = pd.concat([class_rows] * n_to_generate, ignore_index=True)
            neighbor_rows = synthetic.copy()
        else:
            base_indices = rng.integers(0, len(class_rows), size=n_to_generate)
            base_rows = class_rows.iloc[base_indices].reset_index(drop=True)

            if numeric_columns:
                n_neighbors = min(k_neighbors + 1, len(class_rows))
                nn = NearestNeighbors(n_neighbors=n_neighbors)
                nn.fit(class_rows[numeric_columns])
                neighbor_matrix = nn.kneighbors(
                    class_rows[numeric_columns],
                    return_distance=False,
                )[:, 1:]
                if neighbor_matrix.shape[1] == 0:
                    neighbor_indices = base_indices
                else:
                    neighbor_choice = rng.integers(0, neighbor_matrix.shape[1], size=n_to_generate)
                    neighbor_indices = neighbor_matrix[base_indices, neighbor_choice]

                neighbor_rows = class_rows.iloc[neighbor_indices].reset_index(drop=True)
                synthetic = base_rows.copy()
                step = rng.random((n_to_generate, len(numeric_columns)))
                synthetic[numeric_columns] = base_rows[numeric_columns].to_numpy() + step * (
                    neighbor_rows[numeric_columns].to_numpy() - base_rows[numeric_columns].to_numpy()
                )
            else:
                synthetic = base_rows.copy()
                neighbor_rows = class_rows.iloc[
                    rng.integers(0, len(class_rows), size=n_to_generate)
                ].reset_index(drop=True)

            for column in categorical_columns:
                pick_base = rng.random(n_to_generate) < 0.5
                synthetic[column] = np.where(pick_base, base_rows[column], neighbor_rows[column])

        generated_batches.append(synthetic)
        generated_labels.append(pd.Series([class_label] * n_to_generate, name=y_series.name))

    if generated_batches:
        X_resampled = pd.concat([X_df] + generated_batches, ignore_index=True)
        y_resampled = pd.concat([y_series] + generated_labels, ignore_index=True)
        order = rng.permutation(len(y_resampled))
        X_resampled = X_resampled.iloc[order].reset_index(drop=True)
        y_resampled = y_resampled.iloc[order].reset_index(drop=True)
    else:
        X_resampled = X_df
        y_resampled = y_series

    return X_resampled, y_resampled


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


class FoldSafeSmoteLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    LightGBM 二分类封装器。

    作用：
    - 在每个 `fit` 调用内部重新学习缺失值填充值与标准化器，避免 CV 泄漏。
    - 在当前训练折内部再切一小块验证集，专门用于 early stopping。
    - 支持在训练折内二选一地使用 SMOTE 或 `scale_pos_weight` 平衡类别。
    - 统一清洗 `RETENTION_TIME`，把类似 `17.9 and 18.5` 的多值文本解析为均值。

    关键超参数：
    - `random_state`: 统一控制数据处理、SMOTE 与 LightGBM 的随机性。
    - `device_type`: 训练设备，项目强制限定为 `cuda`。
    - `model_n_jobs`: LightGBM 单模型内部线程数。
    - `balance_strategy`: 类别平衡策略，可选 `smote` 或 `scale_pos_weight`。
    - `smote_k_neighbors`: 折内 SMOTE 的近邻数。
    - `scale_pos_weight`: 当使用 `scale_pos_weight` 时传给 LightGBM 的正类权重；为空则按折内训练子集自动计算。
    - `early_stopping_rounds`: 折内 early stopping 的 patience；设为 0 表示禁用。
    - `early_stopping_validation_fraction`: 从当前训练折中切出的验证集比例。
    - `num_leaves`: 单棵树的最大叶子数，越大越容易拟合复杂模式。
    - `learning_rate`: 每轮 boosting 的步长，越小通常越稳。
    - `n_estimators`: boosting 轮数，通常与 `learning_rate` 联动。
    - `max_depth`: 树深上限，用于限制树结构复杂度。
    - `subsample`: 行采样比例，降低同一批样本反复参与建树的风险。
    - `colsample_bytree`: 列采样比例，降低特征共适应。
    - `min_child_samples`: 叶子最少样本数，提高分裂保守性。
    - `min_split_gain`: 节点继续分裂所需的最小增益。
    - `reg_alpha`: L1 正则强度，鼓励更稀疏的分裂模式。
    - `reg_lambda`: L2 正则强度，抑制权重过大。
    """

    def __init__(
        self,
        random_state: int = 42,
        device_type: str = "cuda",
        model_n_jobs: int = 1,
        balance_strategy: str = "smote",
        smote_k_neighbors: int = 5,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 100,
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
        self.device_type = device_type
        self.model_n_jobs = model_n_jobs
        self.balance_strategy = balance_strategy
        self.smote_k_neighbors = smote_k_neighbors
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

    def _resolve_balance_strategy(self) -> str:
        resolved = str(self.balance_strategy).strip().lower()
        if resolved not in {"smote", "scale_pos_weight"}:
            raise ValueError("balance_strategy 只支持 'smote' 或 'scale_pos_weight'。")
        return resolved

    def _resolve_training_scale_pos_weight(self, y_train: pd.Series) -> float:
        if self.scale_pos_weight is not None:
            resolved = float(self.scale_pos_weight)
            if resolved <= 0:
                raise ValueError("scale_pos_weight 必须为正数。")
            return resolved

        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())
        if positive_count <= 0:
            raise ValueError("训练数据中缺少正类样本，无法自动计算 scale_pos_weight。")
        return float(negative_count / positive_count)

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
        validate_cuda_only_requested(self.device_type)

        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            device_type="cuda",
            random_state=self.random_state,
            n_jobs=self.model_n_jobs,
            subsample_freq=1,
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
        if self.early_stopping_rounds < 0:
            raise ValueError("early_stopping_rounds 不能为负数。")

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = y.copy() if isinstance(y, pd.Series) else pd.Series(y, name="target")
        y_series = y_series.reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        self.fit_class_counts_ = y_series.value_counts().sort_index().to_dict()
        self.balance_strategy_ = self._resolve_balance_strategy()

        X_fit, X_eval, y_fit, y_eval = self._split_early_stopping_validation(X_df, y_series)
        self.train_split_class_counts_ = y_fit.value_counts().sort_index().to_dict()
        self.eval_split_class_counts_ = y_eval.value_counts().sort_index().to_dict()

        self.preprocessor_bundle_ = _fit_lightgbm_preprocessor(X_fit)
        X_fit_processed = transform_lightgbm_features(X_fit, self.preprocessor_bundle_)
        X_eval_processed = None
        if not X_eval.empty:
            X_eval_processed = transform_lightgbm_features(X_eval, self.preprocessor_bundle_)

        X_model_fit = X_fit_processed
        y_model_fit = y_fit
        self.effective_scale_pos_weight_ = 1.0
        if self.balance_strategy_ == "smote":
            X_model_fit, y_model_fit = notebook_smote_resample(
                X_fit_processed,
                y_fit,
                random_state=self.random_state,
                k_neighbors=self.smote_k_neighbors,
            )
        else:
            self.effective_scale_pos_weight_ = self._resolve_training_scale_pos_weight(y_fit)

        self.model_fit_class_counts_ = y_model_fit.value_counts().sort_index().to_dict()
        self.resampled_class_counts_ = self.model_fit_class_counts_
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

        self.model_.fit(X_model_fit, y_model_fit, **fit_kwargs)
        self.classes_ = getattr(self.model_, "classes_", np.sort(y_series.unique()))
        self.booster_ = self.model_.booster_
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)

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
        frame.loc[:, RETENTION_TIME_COLUMN] = cleaned_retention_time

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
    if device_type.strip().lower() != "cuda":
        raise RuntimeError("训练设备被强制限定为 CUDA，项目已禁用 CPU / GPU 的静默 fallback。")


def validate_lightgbm_cuda_build(
    device_type: str = "cuda",
    random_state: int = 42,
) -> str:
    validate_cuda_only_requested(device_type)

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
        device_type="cuda",
        n_estimators=4,
        num_leaves=7,
        max_depth=3,
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
        raise RuntimeError(
            "当前 LightGBM 不能以 CUDA 模式训练。\n"
            f"已检测到 lightgbm=={lgbm_version}，但它不是启用 USE_CUDA=1 的构建。\n"
            "项目已禁用任何静默 fallback。\n"
            f"{get_lightgbm_cuda_installation_notes()}"
        ) from exc

    return getattr(lgb, "__version__", "unknown")


def build_lgbm_classifier(
    random_state: int = 42,
    device_type: str = "cuda",
    model_n_jobs: int = 1,
    balance_strategy: str = "smote",
    smote_k_neighbors: int = 5,
    scale_pos_weight: Optional[float] = None,
    early_stopping_rounds: int = 100,
    early_stopping_validation_fraction: float = 0.15,
):
    return FoldSafeSmoteLGBMClassifier(
        random_state=random_state,
        device_type=device_type,
        model_n_jobs=model_n_jobs,
        balance_strategy=balance_strategy,
        smote_k_neighbors=smote_k_neighbors,
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
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    scoring: Optional[str] = None,
    classification_threshold: float = 0.5,
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
    preprocessor_bundle = {
        "artifacts_version": 1,
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
        "numeric_fill_values": estimator_preprocessor.get(
            "numeric_fill_values",
            prepared.get("numeric_fill_values", {}),
        ),
        "categorical_fill_values": estimator_preprocessor.get(
            "categorical_fill_values",
            prepared.get("categorical_fill_values", {}),
        ),
        "scaler": estimator_preprocessor.get("scaler", prepared.get("scaler")),
        "retention_time_column": estimator_preprocessor.get("retention_time_column"),
        "retention_time_multi_value_strategy": estimator_preprocessor.get(
            "retention_time_multi_value_strategy"
        ),
        "retention_time_diagnostics": estimator_preprocessor.get("retention_time_diagnostics"),
        "classes_": list(getattr(estimator, "classes_", [])),
        "classification_threshold": classification_threshold,
        "balance_strategy": getattr(estimator, "balance_strategy_", None),
        "effective_scale_pos_weight": getattr(estimator, "effective_scale_pos_weight_", None),
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
        "artifacts_version": 1,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "target_column": target_column,
        "data_path": data_path,
        "raw_feature_columns": preprocessor_bundle["raw_feature_columns"],
        "model_feature_columns": preprocessor_bundle["model_feature_columns"],
        "continuous_features": preprocessor_bundle["continuous_features"],
        "discrete_features": preprocessor_bundle["discrete_features"],
        "numeric_fill_values": preprocessor_bundle["numeric_fill_values"],
        "categorical_fill_values": preprocessor_bundle["categorical_fill_values"],
        "retention_time_column": preprocessor_bundle["retention_time_column"],
        "retention_time_multi_value_strategy": preprocessor_bundle[
            "retention_time_multi_value_strategy"
        ],
        "retention_time_diagnostics": preprocessor_bundle["retention_time_diagnostics"],
        "dataset_retention_time_diagnostics": prepared.get("retention_time_diagnostics"),
        "classes_": list(getattr(estimator, "classes_", [])),
        "classification_threshold": classification_threshold,
        "balance_strategy": getattr(estimator, "balance_strategy_", None),
        "effective_scale_pos_weight": getattr(estimator, "effective_scale_pos_weight_", None),
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
