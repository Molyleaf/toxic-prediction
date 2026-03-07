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
* **Smoke Validation Script**: `check_lgbm_cuda_pipeline.py` verifies the training chain on a reduced sample instead of launching a full search.
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
├── check_lgbm_cuda_pipeline.py \# Reduced-scope training smoke test
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

For the training notebook, `lightgbm` must be a **CUDA-enabled** build. A CPU-only build is intentionally rejected during preflight and will stop the notebook / smoke script immediately.

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
* The notebook defaults to a **smoke** configuration so you can validate the pipeline without launching a full search.

Useful environment variables:

```bash
LGBM_NOTEBOOK_RUN_MODE=smoke   # or full
LGBM_SMOKE_SAMPLE_SIZE=1024
LGBM_BAYES_N_ITER=2
LGBM_CV_FOLDS=2
```

Before running the full notebook, use the reduced-scope smoke test:

```bash
python check_lgbm_cuda_pipeline.py
```

What the smoke test covers:

* CSV loading
* feature preprocessing
* notebook-local SMOTE replacement
* CUDA preflight for LightGBM
* a reduced BayesSearchCV training loop

If the installed `lightgbm` package was not compiled with CUDA support, the script and notebook will fail fast with an explicit error instead of silently falling back to CPU.

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
