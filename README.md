# Toxic-Prediction: Genotoxicity Prediction Model

This is a machine learning model trained on Massbank data to predict the genotoxicity of chemical substances. The repository includes the pre-trained model and a web frontend for interaction.

## Overview

The core of this project is a machine learning/deep learning model trained on mass spectrometry data (from Massbank) to assess the potential genotoxicity of chemical compounds. This repository provides not only the pre-trained model files but also a complete web application, including a Python-based backend (likely Flask or FastAPI) and a user interface.

## ✨ Features

* **Genotoxicity Prediction**: Predicts substance genotoxicity based on mass spectrometry data.
* **Pre-trained Model**: Includes a ready-to-use model trained on the Massbank dataset.
* **Web Interface**: Provides a simple and user-friendly frontend for making predictions.
* **Containerized**: Includes a `Dockerfile` for quick and easy deployment using Docker.

## 🚀 Tech Stack

* **Backend**: Python (Inferred from `app.py` and `requirements.txt`, likely Flask / FastAPI)
* **Frontend**: HTML, CSS, JavaScript (Located in `static` and `templates`)
* **Model**: (Located in `models`, e.g., Scikit-learn, TensorFlow, PyTorch)
* **Deployment**: Docker

## 📂 Project Structure

```

.
├── models/            \# Stores pre-trained model files
├── static/            \# Stores static assets (CSS, JS, images)
├── templates/         \# Stores HTML templates
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

**d. Run the application**

```bash
python app.py
```

After launching, open your browser and navigate to `http://127.0.0.1:5000` (or the port specified in the console) to use the application.

-----

### 2\. Running with Docker

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