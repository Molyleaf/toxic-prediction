import os
import joblib
from catboost import CatBoostClassifier
import pathlib

# --- 配置 ---
MODEL_DIR = 'models'

# 定义 pkl 模型文件列表
MODEL_FILES = [
    'cb_best_1.pkl',
    'cb_best_2.pkl',
    'cb_best_3.pkl',
    'cb_best_4.pkl',
    'cb_best_5.pkl',
]

if __name__ == "__main__":
    print("开始将 .pkl 转换为 .cbm 格式...")

    for pkl_filename in MODEL_FILES:
        pkl_path = os.path.join(MODEL_DIR, pkl_filename)
        cbm_filename = pkl_filename.replace('.pkl', '.cbm')
        cbm_path = os.path.join(MODEL_DIR, cbm_filename)

        if not os.path.exists(pkl_path):
            print(f"[!] 未找到 {pkl_path}，跳过。")
            continue

        print(f"--- 正在转换: {pkl_path} ---")

        try:
            # 1. 加载 Joblib (PKL) 模型
            print(f"    1/2: 正在加载 {pkl_path}...")
            model = joblib.load(pkl_path)
            if not isinstance(model, CatBoostClassifier):
                print(f"[!] {pkl_filename} 不是 CatBoostClassifier。跳过。")
                continue

            # 2. 保存为 CBM 格式
            print(f"    2/2: 正在保存为 {cbm_path}...")
            model.save_model(cbm_path, format="cbm")

            # 3. 报告大小差异
            pkl_size = pathlib.Path(pkl_path).stat().st_size / (1024 * 1024)
            cbm_size = pathlib.Path(cbm_path).stat().st_size / (1024 * 1024)
            print(f"    [✓] 转换成功！")
            print(f"        PKL 体积: {pkl_size:.2f} MB")
            print(f"        CBM 体积: {cbm_size:.2f} MB")
            print(f"        节省: {(pkl_size - cbm_size):.2f} MB")

        except Exception as e:
            print(f"[!] 转换 {pkl_path} 失败: {e}")

    print("--- CBM 转换完成 ---")
    print("现在你可以从 Dockerfile 中移除 .pkl 文件，只保留 .cbm 文件。")