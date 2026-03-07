from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_FILE = "质谱数据汇总_处理后后后2.csv"
DEFAULT_TARGET_COLUMN = "毒性"
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
) -> NotebookRunConfig:
    resolved_random_seed = int(42 if random_seed is None else random_seed)
    resolved_test_size = float(0.2 if test_size is None else test_size)
    resolved_bayes_n_iter = int(24 if bayes_n_iter is None else bayes_n_iter)
    resolved_cv_folds = int(5 if cv_folds is None else cv_folds)

    # GPU 训练与交叉验证并发通常会争抢同一张卡，这里默认串行执行搜索。
    resolved_model_n_jobs = int(1 if model_n_jobs is None else model_n_jobs)
    resolved_search_n_jobs = int(1 if search_n_jobs is None else search_n_jobs)
    resolved_smote_k_neighbors = int(5 if smote_k_neighbors is None else smote_k_neighbors)

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

    return NotebookRunConfig(
        bayes_n_iter=resolved_bayes_n_iter,
        cv_folds=resolved_cv_folds,
        random_seed=resolved_random_seed,
        test_size=resolved_test_size,
        model_n_jobs=resolved_model_n_jobs,
        search_n_jobs=resolved_search_n_jobs,
        smote_k_neighbors=resolved_smote_k_neighbors,
    )


def load_training_dataframe(
    data_path: Path,
    random_seed: int,
) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=random_seed)

    return data.reset_index(drop=True)


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
    frame["RETENTION_TIME"] = pd.to_numeric(frame["RETENTION_TIME"], errors="coerce").astype("float64")

    X = frame.drop([target_column], axis=1)
    y = frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_fill_values = {}
    for column in numeric_features:
        mode = X_train[column].mode(dropna=True)
        numeric_fill_values[column] = mode.iloc[0] if not mode.empty else 0

    categorical_fill_values = {}
    for column in categorical_features:
        mode = X_train[column].mode(dropna=True)
        categorical_fill_values[column] = mode.iloc[0] if not mode.empty else "missing"

    if numeric_features:
        X_train.loc[:, numeric_features] = X_train[numeric_features].fillna(numeric_fill_values)
        X_test.loc[:, numeric_features] = X_test[numeric_features].fillna(numeric_fill_values)

    if categorical_features:
        X_train.loc[:, categorical_features] = X_train[categorical_features].fillna(categorical_fill_values)
        X_test.loc[:, categorical_features] = X_test[categorical_features].fillna(categorical_fill_values)

    continuous_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    discrete_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    scaler = StandardScaler()
    if continuous_features:
        X_train_continuous = scaler.fit_transform(X_train[continuous_features])
        X_test_continuous = scaler.transform(X_test[continuous_features])
        X_train_processed = pd.DataFrame(
            X_train_continuous,
            columns=continuous_features,
            index=X_train.index,
        )
        X_test_processed = pd.DataFrame(
            X_test_continuous,
            columns=continuous_features,
            index=X_test.index,
        )
    else:
        X_train_processed = pd.DataFrame(index=X_train.index)
        X_test_processed = pd.DataFrame(index=X_test.index)

    if discrete_features:
        X_train_processed = pd.concat([X_train_processed, X_train[discrete_features]], axis=1)
        X_test_processed = pd.concat([X_test_processed, X_test[discrete_features]], axis=1)

    if X_train_processed.isna().any().any() or X_test_processed.isna().any().any():
        raise ValueError("预处理后仍存在缺失值，无法继续执行 SMOTE 和模型训练。")

    X_train_resampled, y_train_resampled = notebook_smote_resample(
        X_train_processed,
        y_train,
        random_state=random_state,
        k_neighbors=smote_k_neighbors,
    )

    return {
        "X_train": X_train_resampled,
        "X_test": X_test_processed,
        "y_train": y_train_resampled,
        "y_test": y_test.reset_index(drop=True),
        "raw_feature_columns": X.columns.tolist(),
        "model_feature_columns": X_test_processed.columns.tolist(),
        "scaler": scaler,
        "continuous_features": continuous_features,
        "discrete_features": discrete_features,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
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
):
    validate_cuda_only_requested(device_type)

    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        device_type="cuda",
        random_state=random_state,
        n_jobs=model_n_jobs,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=5.0,
        min_split_gain=0.05,
        verbosity=-1,
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

    preprocessor_bundle = {
        "artifacts_version": 1,
        "target_column": target_column,
        "raw_feature_columns": prepared["raw_feature_columns"],
        "model_feature_columns": prepared["model_feature_columns"],
        "continuous_features": prepared["continuous_features"],
        "discrete_features": prepared["discrete_features"],
        "numeric_fill_values": prepared["numeric_fill_values"],
        "categorical_fill_values": prepared["categorical_fill_values"],
        "scaler": prepared["scaler"],
        "classes_": list(getattr(estimator, "classes_", [])),
        "classification_threshold": classification_threshold,
    }
    joblib.dump(preprocessor_bundle, preprocessor_path)

    manifest = {
        "artifacts_version": 1,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "target_column": target_column,
        "data_path": data_path,
        "raw_feature_columns": prepared["raw_feature_columns"],
        "model_feature_columns": prepared["model_feature_columns"],
        "continuous_features": prepared["continuous_features"],
        "discrete_features": prepared["discrete_features"],
        "numeric_fill_values": prepared["numeric_fill_values"],
        "categorical_fill_values": prepared["categorical_fill_values"],
        "classes_": list(getattr(estimator, "classes_", [])),
        "classification_threshold": classification_threshold,
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
