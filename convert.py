import os
import joblib
import numpy as np
import openvino as ov
from catboost import CatBoostClassifier
# 确保 catboost 已导入，以便 joblib 可以反序列化

# --- 配置 ---
MODEL_DIR = 'models'
OV_MODEL_DIR = 'models/ov'
# 确保 openvino 转换后的模型存放目录存在
os.makedirs(OV_MODEL_DIR, exist_ok=True)

# 定义模型文件列表
MODEL_FILES = [
    'cb_best_1.pkl',
    'cb_best_2.pkl',
    'cb_best_3.pkl',
    'cb_best_4.pkl',
    'cb_best_5.pkl',
]

# 14 个特征，对应 app.py 中的 'feature_order'
INPUT_SHAPE = [1, 14]

# --- 核心修改 ---
# OpenVINO 转换 CatBoost/SKLearn 模型时需要一个示例输入，而不是 input_shape
# 我们创建一个 float32 类型的零矩阵作为示例
EXAMPLE_INPUT = np.zeros(INPUT_SHAPE, dtype=np.float32)
# --- 结束修改 ---


def convert_model(pkl_filename):
    """
    加载 CatBoost pkl 模型并将其转换为 OpenVINO IR 格式。
    """
    pkl_path = os.path.join(MODEL_DIR, pkl_filename)
    if not os.path.exists(pkl_path):
        print(f"[!] 未找到模型文件: {pkl_path}，跳过。")
        return

    # 构造输出路径
    base_name = pkl_filename.replace('.pkl', '')
    xml_path = os.path.join(OV_MODEL_DIR, f"{base_name}.xml")

    print(f"--- 正在转换: {pkl_path} ---")

    # 1. 加载 CatBoost (joblib) 模型
    try:
        model = joblib.load(pkl_path)
        if not isinstance(model, CatBoostClassifier):
            print(f"[!] {pkl_filename} 不是 CatBoostClassifier。跳过。")
            return
    except Exception as e:
        print(f"[!] 加载 {pkl_path} 失败: {e}")
        return

    # 2. 将模型转换为 OpenVINO 格式
    try:
        # --- 核心修改 ---
        # 使用 example_input 替代 input_shape
        ov_model = ov.convert_model(model, example_input=EXAMPLE_INPUT)
        # --- 结束修改 ---

        # 3. 保存 IR 模型 (xml + bin)
        ov.save_model(ov_model, xml_path)
        print(f"[✓] 成功转换并保存至: {xml_path}")
        print(f"    模型输入: {ov_model.inputs}")
        print(f"    模型输出: {ov_model.outputs}")

    except Exception as e:
        print(f"[!] 转换 {pkl_filename} 失败: {e}")

# --- 执行转换 ---
if __name__ == "__main__":
    print("开始将 CatBoost (.pkl) 模型转换为 OpenVINO IR (.xml/.bin)...")
    for fname in MODEL_FILES:
        convert_model(fname)
    print("--- 转换完成 ---")