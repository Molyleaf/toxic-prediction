# Toxic-Prediction 项目说明

本项目包含两条相互独立的能力链路：

- Web 推理链路：`app.py` 提供基于 Flask 的网页服务，当前使用仓库内现成的 CatBoost 推理模型。
- GPU 训练链路：`lightgbm/light_model.ipynb` 负责训练基于质谱特征的 LightGBM 二分类模型，默认使用 `device_type='gpu'`，并允许显式切回 `device_type='cuda'`。

当前仓库已经将 LightGBM 训练流程统一为单一路径：

- 只在每个交叉验证训练折内部执行 Borderline-SMOTE 增样。
- `scale_pos_weight` 只按“重采样后的新标签分布”自动计算。
- 如果某组参数训练出零分裂模型（all-zero feature importance / 常数预测），会直接判定该候选无效。
- 外层验证集、OOF 与最终测试集始终保持原始真实分布。
- 阈值选择直接基于 CV/OOF。
- 模型输出后会进行严格的概率校准，并把校准器一起导出。

## 目录结构

```text
.
├── app.py                          # Flask Web 服务，当前加载 CatBoost 推理模型
├── cuda_training_support.py        # LightGBM CUDA 训练辅助模块
├── lightgbm/
│   ├── light_model.ipynb           # GPU-first 训练 notebook
│   └── 质谱数据汇总_处理后后后2.csv   # 训练数据
├── models/                         # 已保存的推理模型与训练输出
├── static/                         # 前端静态资源
├── templates/                      # 前端模板
├── requirements.txt                # Python 依赖
└── Dockerfile                      # Web 服务容器化配置
```

## 训练链路的当前设计

### 1. 类别不平衡处理

训练链路不再保留 `smote` / `scale_pos_weight` 两条独立路径，也不再跑两套搜索空间。

现在固定采用下面的顺序：

1. 在当前交叉验证训练折内部重新学习缺失值填充与标准化。
2. 再从该训练折中切出一小块 early stopping 验证集，这块验证集不参与任何重采样。
3. 只对该折的训练子集做 Borderline-SMOTE 增样，用来扩训练样本量，尽量减少高维空间中的跨界插值风险。
4. 基于 Borderline-SMOTE 之后的新标签分布自动计算 `scale_pos_weight`。
5. 训练完成后检查 booster 是否产生了有效分裂；零分裂候选会被直接拒收，不允许进入搜索结果。
6. 用原始、未污染的外层验证集做交叉验证评分。

这意味着：

- 不会先对全量数据做 SMOTE 再切 Fold。
- 不会再使用原始训练集比例去设置 `scale_pos_weight`。
- `scale_pos_weight` 才是类别差值的最终平衡手段，SMOTE 只负责把训练集撑大到更有利于学习的规模。

### 2. 超参数搜索方向

新的搜索范围优先做两件事：

- 保留足够的树容量与 boosting 长度
- 优先避免把分裂门槛和正则一开始就抬得过高

目标是先让搜索稳定找到“能分裂、能学习”的候选，再在这个基础上继续比较泛化表现。

Notebook 顶部默认搜索空间如下：

```python
NOTEBOOK_LGBM_SEARCH_SPACES = {
    "num_leaves": Integer(24, 127),
    "learning_rate": Real(1e-2, 8e-2, prior="log-uniform"),
    "n_estimators": Integer(1200, 4500),
    "max_depth": Categorical([4, 5, 6, 7, 8, 9]),
    "subsample": Real(0.75, 1.00),
    "colsample_bytree": Real(0.75, 1.00),
    "min_child_samples": Integer(10, 80),
    "min_split_gain": Real(1e-4, 0.12, prior="log-uniform"),
    "reg_alpha": Real(1e-3, 0.6, prior="log-uniform"),
    "reg_lambda": Real(1e-2, 8.0, prior="log-uniform"),
}
```

同时默认还做了这些调整：

- `NOTEBOOK_BAYES_N_ITER = 48`
- `NOTEBOOK_EARLY_STOPPING_ROUNDS = 300`
- `NOTEBOOK_INITIAL_THRESHOLD = 0.42`
- `BayesSearchCV(error_score=0.0)`，允许个别崩塌候选被跳过而不是整轮中断

### 3. 阈值与概率校准

训练流程现在明确区分三件事：

- 选参：看 CV AUC
- 阈值：看训练集 OOF 上的校准后概率
- 最终泛化表现：看独立测试集

实际顺序如下：

1. `BayesSearchCV` 找到最优超参数。
2. 用同一套最佳参数重新跑一遍训练集 OOF，收集每个样本的折外原始概率。
3. 在 OOF 原始概率上拟合概率校准器。
4. 用校准后的 OOF 概率进行阈值探测与选择。
5. 再把该校准器应用到最终 refit 模型的输出上，对测试集做最终评估。

如果最终 `best_score_ <= 0.5`，notebook 会直接抛出异常并停止，不再继续输出一个实际上不可用的常数模型。

默认校准策略：

- 首选 `Isotonic Regression`
- 如果 OOF 分数离散度或样本量不够，会自动回退到 `Platt Scaling`

导出的推理资产里会同时保存：

- 预处理状态
- 概率校准器
- 阈值选择元数据
- 最终分类阈值

### 4. RTX 4090 训练利用率

LightGBM 的 GPU / CUDA 训练并不是只有 GPU 在工作，CPU 仍然负责一部分直方图构建、数据调度和喂数。

当前默认策略：

- `BayesSearchCV` worker 固定为 `1`，避免一张卡被多个外层搜索 worker 抢占。
- 单个 LightGBM 模型的 `model_n_jobs` 默认根据 CPU 核数自动给到 `4 ~ 16` 个线程，降低 GPU 因 CPU 单线程喂数不足而空转的概率。
- 搜索空间本身也扩大了树容量和 boosting 长度，让 GPU 每轮训练有更稳定的工作量。

## Notebook 使用方式

训练 notebook 在 `lightgbm/light_model.ipynb`。

建议按顺序运行四个代码单元：

1. 导入依赖、加载 `cuda_training_support.py`、设置全局训练参数。
2. 读取 CSV、清洗 `RETENTION_TIME`、切分训练集和测试集、做 CUDA 预检。
3. 跑单路径 `BayesSearchCV`，得到最终 `best_estimator_`。
4. 生成 OOF 原始概率、做概率校准、在校准后的 OOF 上选阈值、评估测试集、导出模型资产。

第三个代码单元现在还会做两层保护：

- 候选模型如果训练后没有任何有效分裂，会被直接判定为崩塌候选。
- 特征重要性绘图前会先检查是否存在非零 importance，避免训练已经失败时在绘图阶段再次报错。

### Notebook 顶部关键参数

```python
NOTEBOOK_DEVICE_TYPE = "gpu"
NOTEBOOK_RANDOM_SEED = 114514
NOTEBOOK_TEST_SIZE = 0.15
NOTEBOOK_BAYES_N_ITER = 48
NOTEBOOK_CV_FOLDS = 5
NOTEBOOK_MODEL_N_JOBS = cuda_training_support.resolve_recommended_model_n_jobs()
NOTEBOOK_SEARCH_N_JOBS = 1
NOTEBOOK_SMOTE_K_NEIGHBORS = 3
NOTEBOOK_SMOTE_SAMPLING_STRATEGY = 0.75
NOTEBOOK_EARLY_STOPPING_ROUNDS = 300
NOTEBOOK_EARLY_STOPPING_VALIDATION_FRACTION = 0.15
NOTEBOOK_BAYES_SCORING = "roc_auc"
NOTEBOOK_INITIAL_THRESHOLD = 0.42
NOTEBOOK_THRESHOLD_SELECTION_METRIC = "F1"
NOTEBOOK_CALIBRATION_METHOD = "isotonic"
```

## 导出产物

训练完成后会输出三个文件：

- `models/lightgbm_cuda_model.txt`
  - 原生 LightGBM booster 文件
- `models/lightgbm_cuda_preprocessor.joblib`
  - 预处理器、`RETENTION_TIME` 清洗信息、概率校准器、阈值元数据、重采样信息
- `models/lightgbm_cuda_inference_assets.json`
  - 适合人工查看的清单文件，记录模型路径、阈值、概率校准信息、重采样信息与训练配置

虽然默认训练设备已切到 `gpu`（OpenCL），当前 notebook 仍沿用历史输出文件名 `lightgbm_cuda_*`，避免影响现有资产引用路径。

## 环境安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

训练 notebook 现在额外依赖 `imbalanced-learn`，用于折内 Borderline-SMOTE。

为避免 `numpy.dtype size changed` 这类 ABI 报错，`requirements.txt` 已固定以下核心版本：

- `numpy==2.2.6`
- `pandas==2.2.3`
- `scikit-learn==1.6.1`
- `imbalanced-learn>=0.14,<0.15`

如果你的 Linux / Python 3.10 环境之前已经装过其他版本，建议先把科学计算栈强制重装一遍：

```bash
python -m pip uninstall -y numpy pandas scipy scikit-learn imbalanced-learn
python -m pip install --no-cache-dir --force-reinstall \
  "numpy==2.2.6" \
  "pandas==2.2.3" \
  "scikit-learn==1.6.1" \
  "imbalanced-learn>=0.14,<0.15"
python -m pip install -r requirements.txt
```

如果你看到下面这类错误：

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

这通常说明：

- 你先装了某个版本的 `pandas` / `scikit-learn`
- 后面又单独升级或降级了 `numpy`
- 旧 wheel 仍然链接着另一套 `numpy` ABI

这时不要只重装 `pandas`，而是按上面的命令把整套科学计算依赖一起重装。

### 2. 安装 GPU 版 LightGBM

项目训练链路固定要求加速设备为 `gpu` 或 `cuda`，不会静默回退到 CPU。

当前 notebook 默认先尝试 `device_type='gpu'`。如果你确认当前 CUDA 路径稳定，也可以把 `NOTEBOOK_DEVICE_TYPE` 显式改回 `cuda`。

推荐安装方式：

```bash
pip uninstall -y lightgbm
pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON
```

如果需要先编译源码：

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
cmake -B build -S . -DUSE_GPU=ON
cmake --build build -j4
```

默认 `gpu` 路径至少需要：

- Windows 或 Linux 上可用的 OpenCL 运行时 / 显卡驱动
- CMake
- GCC、Clang 或 MSVC 中的一种可用编译器

如果你改回 `device_type='cuda'`，则仍需要：

- Linux 或 WSL2
- CMake 3.28 或更高版本
- GCC 或 Clang
- CUDA Toolkit 11.0 或更新版本

## Web 服务运行

如果只需要启动现有网页推理服务：

```bash
python app.py
```

默认会在本地启动 Flask 服务，并使用仓库内已有的 CatBoost 推理模型。

## Docker

如果只运行 Web 服务，也可以使用 Docker：

```bash
docker build -t toxic-prediction .
docker run -p 5000:5000 toxic-prediction
```

## 注意事项

- 训练 notebook 与 Web 推理服务使用的模型不是同一条链路；Web 端当前仍然加载 CatBoost 产物。
- 训练 notebook 会在启动时先做 CUDA 预检；如果当前 `lightgbm` 不是 CUDA 构建，会直接报错退出。
- 阈值选择和概率校准都只参考训练集 OOF，不会使用测试集反向调参。
- 如果想修改训练行为，优先改 notebook 顶部的 `NOTEBOOK_*` 全局变量，再从头重新运行整个 notebook。
