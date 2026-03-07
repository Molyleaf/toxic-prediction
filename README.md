# Toxic-Prediction: Genotoxicity Prediction Model

This is a machine learning project for predicting the genotoxicity of chemical substances from Massbank-derived features. The repository includes a pre-trained inference model, a web frontend, and a CUDA-only LightGBM training notebook.

## Overview

The core of this project is a mass-spectrometry-based toxicity classifier. The repository contains two separate paths:

* **Inference path**: a Flask web application that serves a pre-trained CatBoost model.
* **Training path**: a LightGBM notebook under `lightgbm/light_model.ipynb` that now runs in **CUDA-only** mode and refuses any silent CPU fallback.

## ✨ Features

* **Genotoxicity Prediction**: Predicts substance genotoxicity based on mass spectrometry data.
* **Pre-trained Model**: Includes a ready-to-use model trained on the Massbank dataset.
* **CUDA-only Training Notebook**: LightGBM training is pinned to `device_type='cuda'` and performs a preflight check before any real training starts.
* **Leak-free CV Pipeline**: missing-value filling, scaling, and SMOTE are learned inside each cross-validation training fold instead of before BayesSearchCV.
* **Notebook-first Training Flow**: the notebook is organized into config/imports, pre-training diagnostics, and a single training-and-save block.
* **Robust `RETENTION_TIME` Cleaning**: multi-value text such as `17.9 and 18.5` is parsed into a single numeric value instead of being dropped as `NaN`.
* **Notebook Progress Bar**: BayesSearchCV progress is shown in the notebook through `tqdm.auto`.
* **Native LightGBM Export**: the best trained booster is saved to `models/lightgbm_cuda_model.txt`.
* **Inference Artifact Bundle**: preprocessing state is saved alongside the model in `models/lightgbm_cuda_preprocessor.joblib` and `models/lightgbm_cuda_inference_assets.json`.
* **Web Interface**: Provides a simple and user-friendly frontend for making predictions.
* **Containerized**: Includes a `Dockerfile` for quick and easy deployment using Docker.

## 🚀 Tech Stack

* **Backend**: Python (Inferred from `app.py` and `requirements.txt`, likely Flask / FastAPI)
* **Frontend**: HTML, CSS, JavaScript (Located in `static` and `templates`)
* **Training Model**: LightGBM (`lightgbm/light_model.ipynb`, CUDA-only)
* **Inference Model**: CatBoost (`models/`)
* **Deployment**: Docker

## 📂 Project Structure

```

.
├── lightgbm/          \# Training notebook and training dataset
├── models/            \# Stores pre-trained inference model files
├── static/            \# Stores static assets (CSS, JS, images)
├── templates/         \# Stores HTML templates
├── cuda_training_support.py   \# Shared CUDA-only training helpers
├── .idea/             \# IDE configuration (can be ignored)
├── app.py             \# Main application backend script
├── Dockerfile         \# Docker configuration file
├── README.md          \# This README file
└── requirements.txt   \# Python dependency list

````

## 🛠️ Getting Started

### 1. Local Setup (Virtual Environment Recommended)

**a. Clone the repository**
```bash
git clone [https://github.com/Molyleaf/toxic-prediction.git](https://github.com/Molyleaf/toxic-prediction.git)
cd toxic-prediction
````

**b. (Optional) Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**c. Install dependencies**

```bash
pip install -r requirements.txt
```

For the training notebook, `lightgbm` must be a **CUDA-enabled** build. A CPU-only build is intentionally rejected during preflight and will stop the notebook immediately.

### Installing CUDA LightGBM on Linux

The notebook uses `device_type='cuda'`, which means the plain Windows wheel is not sufficient. Keep the web app on Windows if you want, but move the training notebook to Linux or WSL2 and reinstall LightGBM from source with CUDA enabled.

Recommended Python package install:

```bash
pip uninstall -y lightgbm
pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON
```

If you prefer compiling LightGBM first:

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
cmake -B build -S . -DUSE_CUDA=ON
cmake --build build -j4
```

Prerequisites from the official LightGBM installation guide:

* Linux
* CMake 3.28 or newer
* GCC or Clang
* CUDA Toolkit 11.0 or newer

**d. Run the application**

```bash
python app.py
```

After launching, open your browser and navigate to `http://127.0.0.1:5000` (or the port specified in the console) to use the application.

### 2. CUDA Training Notebook

The training notebook is located at `lightgbm/light_model.ipynb`.

Key behavior:

* Training is locked to `cuda`.
* CPU fallback is disabled on purpose.
* The first code cell centralizes imports, path fixes, `NOTEBOOK_*` globals, the commented LightGBM search space, and the output paths for all inference artifacts.
* `NOTEBOOK_CONFIG` is built from those globals, so edit that first cell and rerun the notebook from the top when you want to change training behavior.
* The second code cell runs data loading, `RETENTION_TIME` cleaning diagnostics, and train/test splitting before any fitting starts.
* The third code cell runs the only BayesSearchCV training pass, shows a `tqdm.auto` progress bar, evaluates the best estimator, and saves the full inference artifact bundle.
* Missing-value filling, scaling, and SMOTE happen inside the estimator `fit`, so each CV training fold learns its own preprocessing state and performs its own SMOTE pass.

Main globals in the first notebook cell:

```python
# NOTEBOOK_RANDOM_SEED: controls the random seed for the split, BayesSearchCV, SMOTE, and LightGBM.
NOTEBOOK_RANDOM_SEED = 114514
# NOTEBOOK_TEST_SIZE: fraction reserved for the final hold-out test set.
NOTEBOOK_TEST_SIZE = 0.25
# NOTEBOOK_BAYES_N_ITER: number of candidate parameter sets sampled by BayesSearchCV.
NOTEBOOK_BAYES_N_ITER = 24
# NOTEBOOK_CV_FOLDS: number of stratified cross-validation folds.
NOTEBOOK_CV_FOLDS = 5
# NOTEBOOK_MODEL_N_JOBS: thread count used inside a single LightGBM fit.
NOTEBOOK_MODEL_N_JOBS = 1
# NOTEBOOK_SEARCH_N_JOBS: BayesSearchCV worker count; `1` is safer for a shared GPU.
NOTEBOOK_SEARCH_N_JOBS = 1
# NOTEBOOK_SMOTE_K_NEIGHBORS: nearest-neighbor count used by fold-local SMOTE.
NOTEBOOK_SMOTE_K_NEIGHBORS = 5
# NOTEBOOK_BAYES_SCORING: model-selection metric used by BayesSearchCV.
NOTEBOOK_BAYES_SCORING = "roc_auc"
# NOTEBOOK_BAYES_VERBOSE: BayesSearchCV verbosity level.
NOTEBOOK_BAYES_VERBOSE = 0
```

The same top cell also defines `NOTEBOOK_LGBM_SEARCH_SPACES`, with every search dimension annotated in-place:

```python
NOTEBOOK_LGBM_SEARCH_SPACES = {
    "num_leaves": Integer(8, 31),  # max leaves per tree; smaller is more conservative.
    "learning_rate": Real(3e-3, 8e-3, prior="log-uniform"),  # boosting step size.
    "n_estimators": Integer(500, 1800),  # number of boosting rounds.
    "max_depth": Categorical([3, 4, 5, 6]),  # hard tree depth cap.
    "subsample": Real(0.6, 0.85),  # row sampling ratio per tree.
    "colsample_bytree": Real(0.6, 0.85),  # feature sampling ratio per tree.
    "min_child_samples": Integer(50, 200),  # minimum samples required in a leaf.
    "min_split_gain": Real(1e-2, 0.3, prior="log-uniform"),  # minimum gain required to split.
    "reg_alpha": Real(0.1, 10.0, prior="log-uniform"),  # L1 regularization strength.
    "reg_lambda": Real(5.0, 50.0, prior="log-uniform"),  # L2 regularization strength.
}
```

`RETENTION_TIME` handling is also centralized: plain numeric values stay unchanged, multi-value text rows are reduced to the mean of all numeric tokens, and truly missing values remain missing until they are imputed with the training-fold mode.

The notebook training flow covers:

* CSV loading
* `RETENTION_TIME` normalization and diagnostics
* training diagnostics before fitting
* fold-local preprocessing learned independently inside every CV training fold
* fold-local SMOTE replacement applied only to the current training fold
* CUDA preflight for LightGBM
* a full BayesSearchCV training loop with stronger regularization to reduce overfitting
* notebook progress tracking through `tqdm.auto`
* exporting the best booster to `models/lightgbm_cuda_model.txt`
* exporting preprocessing state to `models/lightgbm_cuda_preprocessor.joblib`
* exporting an inference manifest to `models/lightgbm_cuda_inference_assets.json`

The downstream inference bundle consists of:

* `models/lightgbm_cuda_model.txt`: native LightGBM booster
* `models/lightgbm_cuda_preprocessor.joblib`: fitted scaler, fill values, feature grouping, feature order, and `RETENTION_TIME` cleaning metadata
* `models/lightgbm_cuda_inference_assets.json`: human-readable manifest of the saved inference assets

If the installed `lightgbm` package was not compiled with CUDA support, the notebook will fail fast with an explicit error instead of silently falling back to CPU.

### 3\. Running with Docker

If you have Docker installed, you can run the project with these commands.

**a. Build the Docker image**

```bash
docker build -t toxic-prediction .
```

**b. Run the Docker container**

```bash
# This maps port 5000 on your host to port 5000 in the container
docker run -p 5000:5000 toxic-prediction
```

The application will be available at `http://localhost:5000`.

## 🤝 Contributing

Contributions are welcome\! If you have suggestions or want to contribute code, please follow these steps:

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License.
