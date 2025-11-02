from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from catboost import CatBoostClassifier

# --- 初始化 Flask App ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['SECRET_KEY'] = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 统一阈值 ---
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", 0.47))

# --- 模型与Scaler缓存 ---
MODELS = {}
SCALERS = {}

# --- 核心修改：模型映射现在指向 .cbm 文件 ---
MODEL_MAPPING = {
    'cb_1': {'model': 'models/cb_best_1.cbm', 'scaler': 'models/scaler_1.joblib', 'name': 'Model 1'},
    'cb_2': {'model': 'models/cb_best_2.cbm', 'scaler': 'models/scaler_2.joblib', 'name': 'Model 2'},
    'cb_3': {'model': 'models/cb_best_3.cbm', 'scaler': 'models/scaler_3.joblib', 'name': 'Model 3'},
    'cb_4': {'model': 'models/cb_best_4.cbm', 'scaler': 'models/scaler_4.joblib', 'name': 'Model 4'},
    'cb_5': {'model': 'models/cb_best_5.cbm', 'scaler': 'models/scaler_5.joblib', 'name': 'Model 5'},
}

def get_model_by_choice(choice_key: str):
    """
    根据选择返回 CatBoost 模型对象 (从 .cbm 加载)，并缓存。
    """
    if choice_key not in MODEL_MAPPING:
        raise ValueError("无效的模型选择。")

    if choice_key in MODELS:
        return MODELS[choice_key]

    cbm_path = MODEL_MAPPING[choice_key]['model']

    # --- 核心修改：使用 CatBoost 加载 CBM ---
    if not os.path.exists(cbm_path):
        raise FileNotFoundError(f"未找到 CBM 模型文件: {cbm_path}。请先运行 convert_to_cbm.py 脚本。")

    try:
        model_obj = CatBoostClassifier()  # 1. 创建一个空模型
        model_obj.load_model(cbm_path)    # 2. 从文件加载

        MODELS[choice_key] = model_obj
        print(f"成功加载 CBM 模型: {cbm_path}")
        return model_obj
    except Exception as e:
        print(f"加载 CBM 模型 {cbm_path} 失败: {e}")
        raise e
    # --- 结束修改 ---


def get_scaler_by_choice(choice_key: str):
    """
    根据选择返回对应的标准化器（scaler），并缓存。
    """
    if choice_key not in MODEL_MAPPING:
        raise ValueError("无效的模型选择。")

    if choice_key in SCALERS:
        return SCALERS[choice_key]

    scaler_file = MODEL_MAPPING[choice_key]['scaler']
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"未找到scaler文件: {scaler_file}。")

    scaler_obj = joblib.load(scaler_file)
    SCALERS[choice_key] = scaler_obj
    return scaler_obj


# --- 工具函数：自动截断表头以上的行，并设置表头 ---
def load_excel_with_header_cleanup(path):
    # ... (此函数无需修改) ...
    try:
        raw = pd.read_excel(path, header=None)
    except Exception as read_err:
        raise ValueError(f"读取Excel失败: {read_err}")

    header_idx = None
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.strip()
        values_lower = set(v.lower() for v in row.values if v and v.lower() != 'nan')
        if 'mass' in values_lower and 'intensity' in values_lower:
            header_idx = i
            break

    if header_idx is not None:
        header_row = raw.iloc[header_idx].astype(str).str.strip().tolist()
        df = raw.iloc[header_idx + 1:].copy()
        df.columns = header_row
        df = df.loc[:, [c for c in df.columns if c and str(c).strip().lower() != 'nan']]
        df = df.reset_index(drop=True)
        return df
    else:
        df = pd.read_excel(path)
        return df


# --- 特征提取函数 (修复：变量小写 & 优化 max 调用) ---
def extract_features_from_df(df):
    # ... (此函数无需修改) ...
    ff1 = df['Mass'].astype(float).tolist()
    ff2 = df['Intensity'].astype(float).tolist()
    ff3 = df['rel.int.'].astype(float).tolist()

    pn = len(df)
    if pn == 0:
        raise ValueError("数据处理后为空，无法提取特征。请检查输入文件。")

    max_ff3 = max(ff3)
    id_val = max_ff3 / pn
    max_index = ff3.index(max_ff3)
    bp_val = ff1[max_index]
    bpp_val = bp_val - ff1[max_index - 1] if max_index > 0 else 0
    max_ff1 = max(ff1)
    max_index_mz = ff1.index(max_ff1)
    max_mp_val = max_ff1 - ff1[max_index_mz - 1] if max_index_mz > 0 else 0
    min_ff1 = min(ff1)
    ff1_average = sum(ff1) / len(ff1)
    ff1_bzc = np.std(ff1)
    ff2_average = sum(ff2) / len(ff2)
    ff2_bzc = np.std(ff2)

    result_dict = {
        'PN': [pn], 'ID': [id_val], 'BP': [bp_val], 'BPP': [bpp_val],
        'MaxM': [max_ff1], 'MaxMP': [max_mp_val], 'MinM': [min_ff1],
        'MM': [ff1_average], 'MSD': [ff1_bzc], 'IM': [ff2_average],
        'ISD': [ff2_bzc]
    }
    return pd.DataFrame(result_dict)


def run_prediction(model, scaled_features):
    """
    统一预测函数 (仅 CatBoost)。
    返回 (prob_nontoxic, prob_toxic)
    """
    # --- 核心修改：移除 OpenVINO 逻辑 ---
    try:
        # model 现在一定是 CatBoost 对象
        proba = model.predict_proba(scaled_features)[0]
        prob_nontoxic = float(proba[0])
        prob_toxic = float(proba[1])
    except (AttributeError, TypeError):
        # 若模型不支持 predict_proba，回退为硬预测
        prediction = model.predict(scaled_features)
        prob_toxic = float(prediction[0])
        prob_nontoxic = 1.0 - prob_toxic
    return prob_nontoxic, prob_toxic
    # --- 结束修改 ---


# --- Flask 路由 ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # ... (此函数无需修改) ...
    if 'file' not in request.files:
        flash('未找到文件部分')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('未选择文件')
        return redirect(url_for('index'))
    try:
        retention_time = float(request.form['retention_time'])
        collision_energy = float(request.form['collision_energy'])
        precursor_type = int(request.form['precursor_type'])
    except (ValueError, TypeError):
        flash('输入的参数格式不正确，请输入数字。')
        return redirect(url_for('index'))
    model_choice = request.form.get('model_choice', 'cb_5')
    valid_choices = ('cb_1', 'cb_2', 'cb_3', 'cb_4', 'cb_5', 'cb_all')
    if model_choice not in valid_choices:
        flash('无效的模型选择。当前支持 CatBoost dataset 1~5 或 All。')
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # --- 2. 数据预处理 ---
            df = load_excel_with_header_cleanup(filepath)
            if 'Intensity' not in df.columns or 'Mass' not in df.columns:
                renamed = {c: str(c).strip() for c in df.columns}
                df = df.rename(columns=renamed)
                cols_lower = {str(c).strip().lower(): c for c in df.columns}
                rename_map = {}
                if 'mass' in cols_lower:
                    rename_map[cols_lower['mass']] = 'Mass'
                if 'intensity' in cols_lower:
                    rename_map[cols_lower['intensity']] = 'Intensity'
                if rename_map:
                    df = df.rename(columns=rename_map)
                if 'Intensity' not in df.columns or 'Mass' not in df.columns:
                    raise ValueError("上传的Excel文件必须包含 'Intensity' 和 'Mass' 列。")

            keep_cols = ['Mass', 'Intensity']
            df = df[keep_cols].copy()
            df['Mass'] = pd.to_numeric(df['Mass'], errors='coerce')
            df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
            df = df.dropna(subset=['Mass', 'Intensity']).reset_index(drop=True)
            df = df[df['Intensity'] != 0].reset_index(drop=True)
            max_intensity = df['Intensity'].max()
            if pd.isna(max_intensity) or max_intensity <= 0:
                raise ValueError("有效的强度数据为空或为非正数，无法继续。")
            threshold = max_intensity * 0.03
            df = df[df['Intensity'] >= threshold].reset_index(drop=True)
            max_value = df['Intensity'].max()
            df['rel.int.'] = (df['Intensity'] / max_value) * 999
            df['rel.int.'] = df['rel.int.'].clip(upper=999).round().astype(int)

            # --- 3. 特征工程 ---
            features_df = extract_features_from_df(df)
            features_df['RETENTION_TIME'] = retention_time
            features_df['COLLISION_ENERGY'] = collision_energy
            features_df['PRECURSOR_TYPE'] = precursor_type
            feature_order = [
                'PN', 'ID', 'BP', 'BPP', 'MaxM', 'MaxMP', 'MinM', 'MM', 'MSD',
                'IM', 'ISD', 'RETENTION_TIME', 'COLLISION_ENERGY', 'PRECURSOR_TYPE'
            ]
            features_df = features_df[feature_order]

            # --- 4. 标准化和预测 ---
            if model_choice == 'cb_all':
                results = []
                threshold_to_use = PREDICTION_THRESHOLD
                for ck in ['cb_1', 'cb_2', 'cb_3', 'cb_4', 'cb_5']:
                    scaler = get_scaler_by_choice(ck)
                    scaled_features = scaler.transform(features_df)
                    model = get_model_by_choice(ck)
                    prob_nontoxic, prob_toxic = run_prediction(model, scaled_features)
                    label = "Toxic" if prob_toxic >= threshold_to_use else "Nontoxic"
                    results.append({
                        'model_key': ck,
                        'model_name': MODEL_MAPPING[ck]['name'],
                        'label': label,
                        'proba_toxic': prob_toxic,
                        'proba_nontoxic': prob_nontoxic
                    })
                return render_template('result_all.html', results=results)

            # 单模型路径
            scaler = get_scaler_by_choice(model_choice)
            scaled_features = scaler.transform(features_df)
            model = get_model_by_choice(model_choice)
            prob_nontoxic, prob_toxic = run_prediction(model, scaled_features)

            # --- 5. 返回结果 ---
            threshold_to_use = PREDICTION_THRESHOLD
            is_toxic = prob_toxic >= threshold_to_use
            result_text = "Toxic" if is_toxic else "Nontoxic"
            return render_template(
                'result.html',
                prediction=result_text,
                proba_toxic=prob_toxic,
                proba_nontoxic=prob_nontoxic
            )
        except (ValueError, FileNotFoundError, KeyError) as data_err:
            flash(f"处理文件时发生错误: {data_err}")
            return redirect(url_for('index'))
        except Exception as unknown_err:
            flash(f"发生未知错误: {unknown_err}")
            return redirect(url_for('index'))

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)