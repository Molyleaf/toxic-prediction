import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename

# --- OpenVINO 加速检测 ---
try:
    import openvino as ov
    core = ov.Core()
    openvino_available = True
    print("OpenVINO Runtime 检测成功。将优先使用 OpenVINO 加速。")
except ImportError:
    core = None
    openvino_available = False
    print("未找到 OpenVINO Runtime。将回退到 joblib/CatBoost。")

# --- 初始化 Flask App ---
# 假设模板文件在 'templates' 文件夹, 静态文件在 'static'
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['SECRET_KEY'] = 'supersecretkey'  # 用于flash消息
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 统一阈值 ---
# 从环境变量读取阈值，若未设置，默认为 0.5
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", 0.5))

# --- 模型与Scaler缓存（按选择动态加载） ---
MODELS = {}
SCALERS = {}

# 统一管理模型和Scaler的路径
MODEL_MAPPING = {
    'cb_1': {'model': 'models/cb_best_1.pkl', 'scaler': 'models/scaler_1.joblib', 'name': 'Model 1'},
    'cb_2': {'model': 'models/cb_best_2.pkl', 'scaler': 'models/scaler_2.joblib', 'name': 'Model 2'},
    'cb_3': {'model': 'models/cb_best_3.pkl', 'scaler': 'models/scaler_3.joblib', 'name': 'Model 3'},
    'cb_4': {'model': 'models/cb_best_4.pkl', 'scaler': 'models/scaler_4.joblib', 'name': 'Model 4'},
    'cb_5': {'model': 'models/cb_best_5.pkl', 'scaler': 'models/scaler_5.joblib', 'name': 'Model 5'},
}

def get_model_by_choice(choice_key: str):
    """
    根据选择返回模型对象（优先OpenVINO），并缓存。
    """
    if choice_key not in MODEL_MAPPING:
        raise ValueError("无效的模型选择。")

    if choice_key in MODELS:
        return MODELS[choice_key]

    pkl_path = MODEL_MAPPING[choice_key]['model']

    # 优先尝试加载 OpenVINO IR 模型
    if openvino_available:
        # 从 'models/cb_best_1.pkl' 构造 'models/ov/cb_best_1.xml'
        ov_xml_path = pkl_path.replace("models/", "models/ov/").replace(".pkl", ".xml")
        if os.path.exists(ov_xml_path):
            try:
                ov_model = core.read_model(model=ov_xml_path)
                compiled_model = core.compile_model(model=ov_model, device_name="CPU")
                MODELS[choice_key] = compiled_model
                print(f"成功加载 OpenVINO 模型: {ov_xml_path}")
                return compiled_model
            except Exception as ov_err:
                print(f"加载 OpenVINO 模型 {ov_xml_path} 失败: {ov_err}。回退至 PKL。")
        else:
            print(f"未找到 OpenVINO 模型: {ov_xml_path}。回退至 PKL。")

    # 回退到加载原始 PKL 文件
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到模型文件: {pkl_path}。")

    model_obj = joblib.load(pkl_path)
    MODELS[choice_key] = model_obj
    print(f"成功加载 PKL 模型: {pkl_path}")
    return model_obj


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
    """
    读取 Excel，自动找到包含 'Mass' 与 'Intensity' 的表头行。
    """
    try:
        raw = pd.read_excel(path, header=None)
    except Exception as read_err:
        # 修复：重命名 'e' 避免阴影，使用更具体的异常
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
        # 回退
        df = pd.read_excel(path)
        return df


# --- 特征提取函数 (修复：变量小写 & 优化 max 调用) ---
def extract_features_from_df(df):
    # 修复：变量名改为 snake_case
    ff1 = df['Mass'].astype(float).tolist()
    ff2 = df['Intensity'].astype(float).tolist()
    ff3 = df['rel.int.'].astype(float).tolist()

    pn = len(df)
    if pn == 0:
        raise ValueError("数据处理后为空，无法提取特征。请检查输入文件。")

    # 修复：优化 max 调用，避免冗余计算
    max_ff3 = max(ff3) # 已知 ff3 非空 (pn > 0)
    id_val = max_ff3 / pn

    max_index = ff3.index(max_ff3)
    bp_val = ff1[max_index]  # BP

    bpp_val = bp_val - ff1[max_index - 1] if max_index > 0 else 0  # BPP

    max_ff1 = max(ff1)  # MaxM
    max_index_mz = ff1.index(max_ff1)

    max_mp_val = max_ff1 - ff1[max_index_mz - 1] if max_index_mz > 0 else 0  # MaxMP

    min_ff1 = min(ff1)  # MinM
    ff1_average = sum(ff1) / len(ff1)  # MM
    ff1_bzc = np.std(ff1)  # MSD
    ff2_average = sum(ff2) / len(ff2)  # IM
    ff2_bzc = np.std(ff2)  # ISD

    # 修复：字典的 Key 保持大写 (特征名)，Value 使用小写变量
    result_dict = {
        'PN': [pn], 'ID': [id_val], 'BP': [bp_val], 'BPP': [bpp_val],
        'MaxM': [max_ff1], 'MaxMP': [max_mp_val], 'MinM': [min_ff1],
        'MM': [ff1_average], 'MSD': [ff1_bzc], 'IM': [ff2_average],
        'ISD': [ff2_bzc]
    }
    return pd.DataFrame(result_dict)


def run_prediction(model, scaled_features):
    """
    统一预测函数，自动处理 OpenVINO 或 CatBoost 模型。
    返回 (prob_nontoxic, prob_toxic)
    """
    # 1. OpenVINO 模型路径
    if openvino_available and isinstance(model, ov.CompiledModel):
        try:
            # CatBoost 转换的模型通常有2个输出:
            # outputs[0]: 标签 (shape [1, 1])
            # outputs[1]: 概率 (shape [1, 2])
            # 我们需要概率
            input_tensor = model.inputs[0]
            output_tensor = model.outputs[1]

            # 确保输入是 float32
            scaled_features_f32 = scaled_features.astype(np.float32)

            result = model.infer_new_request({input_tensor: scaled_features_f32})

            # result[output_tensor] 的形状是 [[prob_0, prob_1]]
            proba_array = result[output_tensor][0]
            prob_nontoxic = float(proba_array[0])
            prob_toxic = float(proba_array[1])
            return prob_nontoxic, prob_toxic

        except Exception as ov_predict_err:
            print(f"OpenVINO 预测失败: {ov_predict_err}")
            # 回退（虽然理论上不应该发生）
            return 0.0, 0.0

    # 2. 原始 CatBoost (joblib) 模型路径
    else:
        try:
            proba = model.predict_proba(scaled_features)[0]
            prob_nontoxic = float(proba[0])
            prob_toxic = float(proba[1])
        except (AttributeError, TypeError):
            # 修复：更具体的异常捕获
            # 若模型不支持 predict_proba，回退为硬预测
            prediction = model.predict(scaled_features)
            prob_toxic = float(prediction[0])
            prob_nontoxic = 1.0 - prob_toxic
        return prob_nontoxic, prob_toxic


# --- Flask 路由 ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # --- 1. 获取输入 ---
    if 'file' not in request.files:
        flash('未找到文件部分')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('未选择文件')
        return redirect(url_for('index'))

    # 获取表单中的其他字段
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

        # 修复：将 'except Exception' 替换为更具体的异常
        try:
            # --- 2. 数据预处理 ---
            df = load_excel_with_header_cleanup(filepath)

            # 确保 'Mass' 和 'Intensity' 存在 (不区分大小写)
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

            # 若选择 All
            if model_choice == 'cb_all':
                results = []
                # 修复：使用统一的环境变量阈值
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

            # --- 5. 返回结果 (按阈值判定 Toxic/Nontoxic) ---
            # 修复：使用统一的环境变量阈值
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
            # 修复：捕获更具体的错误，并重命名 'e'
            flash(f"处理文件时发生错误: {data_err}")
            return redirect(url_for('index'))
        except Exception as unknown_err:
            # 修复：捕获未知错误，并重命名 'e'
            flash(f"发生未知错误: {unknown_err}")
            return redirect(url_for('index'))

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)