import gc
import os
import tempfile
import threading
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# --- Flask setup ---
URL_PREFIX = "/genotoxic"
BASE_DIR = Path(__file__).resolve().parent
LGBM_ASSET_DIR = BASE_DIR / "models" / "lgbm"
LGBM_MODEL_PATH = LGBM_ASSET_DIR / "lightgbm_model.txt"
LGBM_PREPROCESSOR_PATH = LGBM_ASSET_DIR / "lightgbm_preprocessor.joblib"

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path=f"{URL_PREFIX}/static",
)
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", tempfile.gettempdir())
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "supersecretkey")
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", 0.47))

# --- Cached runtime assets ---
MODEL = None
PREPROCESSOR_BUNDLE = None

# --- Idle cache release ---
CACHE_RELEASE_TIMER = None
CACHE_TIMEOUT_SECONDS = int(os.environ.get("CACHE_TIMEOUT_SECONDS", 300))


def get_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到 LightGBM 模型文件: {LGBM_MODEL_PATH}")

    MODEL = lgb.Booster(model_file=str(LGBM_MODEL_PATH))
    return MODEL


def get_preprocessor_bundle():
    global PREPROCESSOR_BUNDLE

    if PREPROCESSOR_BUNDLE is not None:
        return PREPROCESSOR_BUNDLE

    if not LGBM_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"未找到 LightGBM 预处理器文件: {LGBM_PREPROCESSOR_PATH}")

    PREPROCESSOR_BUNDLE = joblib.load(LGBM_PREPROCESSOR_PATH)
    if not isinstance(PREPROCESSOR_BUNDLE, dict):
        raise TypeError("LightGBM 预处理器资产格式无效，应为字典。")
    return PREPROCESSOR_BUNDLE


def clear_caches():
    global MODEL, PREPROCESSOR_BUNDLE, CACHE_RELEASE_TIMER

    if MODEL is None and PREPROCESSOR_BUNDLE is None:
        return

    print(f"检测到 {CACHE_TIMEOUT_SECONDS} 秒无活动，正在释放 LightGBM 缓存...")
    MODEL = None
    PREPROCESSOR_BUNDLE = None
    CACHE_RELEASE_TIMER = None
    gc.collect()
    print("LightGBM 缓存已释放。")


def reset_cache_timer():
    global CACHE_RELEASE_TIMER

    if CACHE_RELEASE_TIMER:
        CACHE_RELEASE_TIMER.cancel()

    CACHE_RELEASE_TIMER = threading.Timer(CACHE_TIMEOUT_SECONDS, clear_caches)
    CACHE_RELEASE_TIMER.daemon = True
    CACHE_RELEASE_TIMER.start()


def load_excel_with_header_cleanup(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, header=None)
    except Exception as exc:
        raise ValueError(f"读取 Excel 失败: {exc}") from exc

    header_idx = None
    for idx in range(len(raw)):
        row = raw.iloc[idx].astype(str).str.strip()
        values_lower = {value.lower() for value in row.values if value and value.lower() != "nan"}
        if "mass" in values_lower and "intensity" in values_lower:
            header_idx = idx
            break

    if header_idx is None:
        return pd.read_excel(path)

    header_row = raw.iloc[header_idx].astype(str).str.strip().tolist()
    frame = raw.iloc[header_idx + 1 :].copy()
    frame.columns = header_row
    frame = frame.loc[:, [column for column in frame.columns if str(column).strip().lower() != "nan"]]
    return frame.reset_index(drop=True)


def normalize_spectrum_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(columns={column: str(column).strip() for column in frame.columns})
    columns_lower = {str(column).strip().lower(): column for column in normalized.columns}
    rename_map = {}
    if "mass" in columns_lower:
        rename_map[columns_lower["mass"]] = "Mass"
    if "intensity" in columns_lower:
        rename_map[columns_lower["intensity"]] = "Intensity"
    return normalized.rename(columns=rename_map)


def prepare_spectrum_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_spectrum_columns(frame)
    if "Mass" not in frame.columns or "Intensity" not in frame.columns:
        raise ValueError("上传的 Excel 文件必须包含 'Mass' 和 'Intensity' 列。")

    spectrum = frame.loc[:, ["Mass", "Intensity"]].copy()
    spectrum["Mass"] = pd.to_numeric(spectrum["Mass"], errors="coerce")
    spectrum["Intensity"] = pd.to_numeric(spectrum["Intensity"], errors="coerce")
    spectrum = spectrum.dropna(subset=["Mass", "Intensity"]).reset_index(drop=True)
    spectrum = spectrum[spectrum["Intensity"] != 0].reset_index(drop=True)
    if spectrum.empty:
        raise ValueError("有效的质谱峰为空，无法继续。")

    max_intensity = spectrum["Intensity"].max()
    if pd.isna(max_intensity) or max_intensity <= 0:
        raise ValueError("有效的强度数据为空或为非正数，无法继续。")

    threshold = max_intensity * 0.03
    spectrum = spectrum[spectrum["Intensity"] >= threshold].reset_index(drop=True)
    if spectrum.empty:
        raise ValueError("按 3% 最大强度阈值过滤后无有效质谱峰。")

    max_value = spectrum["Intensity"].max()
    spectrum["rel.int."] = (spectrum["Intensity"] / max_value) * 999
    spectrum["rel.int."] = spectrum["rel.int."].clip(upper=999).round().astype(int)
    return spectrum


def extract_features_from_df(frame: pd.DataFrame) -> pd.DataFrame:
    masses = frame["Mass"].astype(float).tolist()
    intensities = frame["Intensity"].astype(float).tolist()
    relative_intensities = frame["rel.int."].astype(float).tolist()

    peak_count = len(frame)
    if peak_count == 0:
        raise ValueError("数据处理后为空，无法提取特征。请检查输入文件。")

    max_relative_intensity = max(relative_intensities)
    intensity_density = max_relative_intensity / peak_count
    base_peak_index = relative_intensities.index(max_relative_intensity)
    base_peak = masses[base_peak_index]
    base_peak_spacing = base_peak - masses[base_peak_index - 1] if base_peak_index > 0 else 0.0

    max_mass = max(masses)
    max_mass_index = masses.index(max_mass)
    max_mass_spacing = max_mass - masses[max_mass_index - 1] if max_mass_index > 0 else 0.0

    result = {
        "PN": [peak_count],
        "ID": [intensity_density],
        "BP": [base_peak],
        "BPP": [base_peak_spacing],
        "MaxM": [max_mass],
        "MaxMP": [max_mass_spacing],
        "MinM": [min(masses)],
        "MM": [sum(masses) / len(masses)],
        "MSD": [np.std(masses)],
        "IM": [sum(intensities) / len(intensities)],
        "ISD": [np.std(intensities)],
    }
    return pd.DataFrame(result)


def build_raw_feature_frame(
    spectrum_features: pd.DataFrame,
    retention_time: float,
    collision_energy: float,
    precursor_type: float,
) -> pd.DataFrame:
    frame = spectrum_features.copy()
    frame["RETENTION_TIME"] = retention_time
    frame["COLLISION_ENERGY"] = collision_energy
    frame["PRECURSOR_TYPE"] = precursor_type
    return frame


def transform_lightgbm_features(raw_features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    raw_input_columns = bundle.get("raw_input_columns") or bundle.get("feature_order")
    feature_order = bundle.get("feature_order")
    scaler = bundle.get("scaler")

    if not raw_input_columns or not feature_order or scaler is None:
        raise ValueError("LightGBM 预处理器资产缺少必要字段。")

    missing_columns = [column for column in raw_input_columns if column not in raw_features.columns]
    if missing_columns:
        raise KeyError(f"输入特征缺少列: {missing_columns}")

    frame = raw_features.loc[:, raw_input_columns].copy()

    for column in bundle.get("retention_time_cast_columns", []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    numeric_impute_values = bundle.get("numeric_impute_values", {})
    numeric_columns = [column for column in raw_input_columns if column in numeric_impute_values]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(numeric_impute_values[column])

    continuous_features = bundle.get("continuous_features", [])
    discrete_features = bundle.get("discrete_features", [])

    if continuous_features:
        continuous_frame = pd.DataFrame(
            scaler.transform(frame.loc[:, continuous_features]),
            columns=continuous_features,
            index=frame.index,
        )
    else:
        continuous_frame = pd.DataFrame(index=frame.index)

    if discrete_features:
        processed = pd.concat([continuous_frame, frame.loc[:, discrete_features]], axis=1)
    else:
        processed = continuous_frame

    processed = processed.loc[:, feature_order].copy()

    processed_feature_dtypes = bundle.get("processed_feature_dtypes", {})
    for column, dtype_name in processed_feature_dtypes.items():
        if column not in processed.columns:
            continue
        if dtype_name.startswith(("float", "int")):
            processed[column] = processed[column].astype(dtype_name)

    if processed.isna().any().any():
        nan_columns = processed.columns[processed.isna().any()].tolist()
        raise ValueError(f"Notebook 预处理后仍存在 NaN，涉及列: {nan_columns}")

    non_numeric_columns = processed.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_columns:
        raise TypeError(f"Notebook 预处理后仍存在非数值列: {non_numeric_columns}")

    return processed


def predict_with_lightgbm(processed_features: pd.DataFrame) -> tuple[float, float]:
    model = get_model()
    prediction = np.asarray(model.predict(processed_features))

    if prediction.ndim == 1:
        prob_toxic = float(prediction[0])
    elif prediction.ndim == 2 and prediction.shape[1] >= 2:
        prob_toxic = float(prediction[0, 1])
    else:
        raise ValueError(f"无法解析 LightGBM 输出形状: {prediction.shape}")

    prob_toxic = float(np.clip(prob_toxic, 0.0, 1.0))
    prob_nontoxic = 1.0 - prob_toxic
    return prob_nontoxic, prob_toxic


@app.route("/", methods=["GET"])
def root_redirect():
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict_legacy():
    return predict()


@app.route(f"{URL_PREFIX}/", methods=["GET"])
@app.route(f"{URL_PREFIX}", methods=["GET"])
def index():
    return render_template("index.html")


@app.route(f"{URL_PREFIX}/predict", methods=["POST"])
def predict():
    reset_cache_timer()

    if "file" not in request.files:
        flash("未找到文件部分")
        return redirect(url_for("index"))

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        flash("未选择文件")
        return redirect(url_for("index"))

    try:
        retention_time = float(request.form["retention_time"])
        collision_energy = float(request.form["collision_energy"])
        precursor_type = float(request.form["precursor_type"])
    except (KeyError, TypeError, ValueError):
        flash("输入的参数格式不正确，请输入数字。")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded_file.filename) or "upload.xlsx"
    filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
    uploaded_file.save(filepath)

    try:
        source_df = load_excel_with_header_cleanup(filepath)
        spectrum_df = prepare_spectrum_dataframe(source_df)
        spectrum_features = extract_features_from_df(spectrum_df)
        raw_feature_frame = build_raw_feature_frame(
            spectrum_features=spectrum_features,
            retention_time=retention_time,
            collision_energy=collision_energy,
            precursor_type=precursor_type,
        )
        processed_features = transform_lightgbm_features(
            raw_features=raw_feature_frame,
            bundle=get_preprocessor_bundle(),
        )
        prob_nontoxic, prob_toxic = predict_with_lightgbm(processed_features)

        result_text = "Toxic" if prob_toxic >= PREDICTION_THRESHOLD else "Nontoxic"
        return render_template(
            "result.html",
            prediction=result_text,
            proba_toxic=prob_toxic,
            proba_nontoxic=prob_nontoxic,
        )
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        flash(f"处理文件时发生错误: {exc}")
        return redirect(url_for("index"))
    except Exception as exc:
        flash(f"发生未知错误: {exc}")
        return redirect(url_for("index"))
    finally:
        try:
            filepath.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
