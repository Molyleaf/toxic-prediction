import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename

# --- 初始化 Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'  # 用于flash消息
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 模型与Scaler缓存（按选择动态加载） ---
MODELS = {}
SCALERS = {}


def get_model_by_choice(choice_key: str):
    """
    根据前端选择返回对应模型对象，并做简单缓存。
    choice_key: 'cb_1' ~ 'cb_5'
    """
    mapping = {
        'cb_1': ('models/cb_best_1.pkl', 'Model 1'),
        'cb_2': ('models/cb_best_2.pkl', 'Model 2'),
        'cb_3': ('models/cb_best_3.pkl', 'Model 3'),
        'cb_4': ('models/cb_best_4.pkl', 'Model 4'),
        'cb_5': ('models/cb_best_5.pkl', 'Model 5'),
    }
    if choice_key not in mapping:
        raise ValueError("无效的模型选择。")

    filename, _ = mapping[choice_key]
    if choice_key in MODELS:
        return MODELS[choice_key]

    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到模型文件: {filename}。")

    model_obj = joblib.load(filename)
    MODELS[choice_key] = model_obj
    return model_obj


def get_scaler_by_choice(choice_key: str):
    """
    根据前端选择返回对应的标准化器（scaler），并做简单缓存。
    choice_key: 'cb_1' ~ 'cb_5'
    """
    mapping = {
        'cb_1': 'models/scaler_1.joblib',
        'cb_2': 'models/scaler_2.joblib',
        'cb_3': 'models/scaler_3.joblib',
        'cb_4': 'models/scaler_4.joblib',
        'cb_5': 'models/scaler_5.joblib',
    }
    if choice_key not in mapping:
        raise ValueError("无效的模型选择。")

    scaler_file = mapping[choice_key]
    if choice_key in SCALERS:
        return SCALERS[choice_key]

    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"未找到scaler文件: {scaler_file}。")

    scaler_obj = joblib.load(scaler_file)
    SCALERS[choice_key] = scaler_obj
    return scaler_obj


# --- 工具函数：自动截断表头以上的行，并设置表头 ---
def load_excel_with_header_cleanup(path):
    """
    读取 Excel，自动找到包含 'Mass' 与 'Intensity' 的表头行：
    - 删除该行以上的所有行
    - 使用该行为列名
    - 返回清洗后的 DataFrame
    若未找到，则回退为常规读取（header=0），并继续后续校验。
    """
    try:
        raw = pd.read_excel(path, header=None)
    except Exception as e:
        raise ValueError(f"读取Excel失败: {e}")

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


# --- 特征提取函数 (源自 Notebook) ---
def extract_features_from_df(df):
    FF1 = df['Mass'].astype(float).tolist()
    FF2 = df['Intensity'].astype(float).tolist()
    FF3 = df['rel.int.'].astype(float).tolist()

    PN = len(df)
    if PN == 0:
        raise ValueError("数据处理后为空，无法提取特征。请检查输入文件。")

    max_FF3 = max(FF3) if FF3 else 0
    ID = max_FF3 / PN

    max_FF3_val = max(FF3)
    max_index = FF3.index(max_FF3_val)
    max_FF3_FF1 = FF1[max_index]  # BP

    cz = max_FF3_FF1 - FF1[max_index - 1] if max_index > 0 else 0  # BPP

    max_FF1 = max(FF1)  # MaxM
    max_index_mz = FF1.index(max_FF1)

    cz2 = max_FF1 - FF1[max_index_mz - 1] if max_index_mz > 0 else 0  # MaxMP

    min_FF1 = min(FF1)  # MinM
    FF1_average = sum(FF1) / len(FF1)  # MM
    FF1_bzc = np.std(FF1)  # MSD
    FF2_average = sum(FF2) / len(FF2)  # IM
    FF2_bzc = np.std(FF2)  # ISD

    result_dict = {
        'PN': [PN], 'ID': [ID], 'BP': [max_FF3_FF1], 'BPP': [cz],
        'MaxM': [max_FF1], 'MaxMP': [cz2], 'MinM': [min_FF1],
        'MM': [FF1_average], 'MSD': [FF1_bzc], 'IM': [FF2_average],
        'ISD': [FF2_bzc]
    }
    return pd.DataFrame(result_dict)


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

    # 获取模型选择（CatBoost 五套数据集 + All）
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
                if 'mass' in cols_lower and 'intensity' in cols_lower:
                    df = df.rename(columns={
                        cols_lower['mass']: 'Mass',
                        cols_lower['intensity']: 'Intensity'
                    })
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
            # 若选择 All，则依次对 cb_1 ~ cb_5 执行并汇总结果
            if model_choice == 'cb_all':
                model_display = {
                    'cb_1': 'Model 1',
                    'cb_2': 'Model 2',
                    'cb_3': 'Model 3',
                    'cb_4': 'Model 4',
                    'cb_5': 'Model 5',
                }
                THRESHOLD = 0.47
                results = []
                for ck in ['cb_1', 'cb_2', 'cb_3', 'cb_4', 'cb_5']:
                    scaler = get_scaler_by_choice(ck)
                    scaled_features = scaler.transform(features_df)

                    model = get_model_by_choice(ck)
                    prediction = model.predict(scaled_features)
                    try:
                        proba = model.predict_proba(scaled_features)[0]
                        prob_nontoxic = float(proba[0])
                        prob_toxic = float(proba[1])
                    except Exception:
                        prob_toxic = float(prediction[0])
                        prob_nontoxic = 1.0 - prob_toxic

                    label = "Toxic" if prob_toxic >= THRESHOLD else "Nontoxic"
                    results.append({
                        'model_key': ck,
                        'model_name': model_display.get(ck, ck),
                        'label': label,
                        'proba_toxic': prob_toxic,
                        'proba_nontoxic': prob_nontoxic
                    })

                return render_template('result_all.html', results=results)

            # 单模型路径保持原有逻辑
            scaler = get_scaler_by_choice(model_choice)
            scaled_features = scaler.transform(features_df)

            model = get_model_by_choice(model_choice)
            prediction = model.predict(scaled_features)
            # 概率输出：有毒(1)与无毒(0)的概率
            try:
                proba = model.predict_proba(scaled_features)[0]
                prob_nontoxic = float(proba[0])
                prob_toxic = float(proba[1])
            except Exception:
                # 若模型不支持 predict_proba，回退为硬预测
                prob_toxic = float(prediction[0])
                prob_nontoxic = 1.0 - prob_toxic
            
            # --- 5. 返回结果（按阈值判定 Toxic/Nontoxic） ---
            THRESHOLD = 0.3  # 概率阈值：>= 0.3 判定为 Toxic
            is_toxic = prob_toxic >= THRESHOLD
            result_text = "Toxic" if is_toxic else "Nontoxic"
            return render_template(
                'result.html',
                prediction=result_text,
                proba_toxic=prob_toxic,
                proba_nontoxic=prob_nontoxic
            )

        except Exception as e:
            flash(f"处理文件时发生错误: {e}")
            return redirect(url_for('index'))

    return redirect(url_for('index'))


if __name__ == '__main__':
    # 允许从网络中的任何IP访问，这是Docker容器所必需的
    app.run(host='0.0.0.0', port=5000, debug=True)