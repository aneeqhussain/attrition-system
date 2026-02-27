# Employee Attrition Prediction System

## Project Overview

The **Employee Attrition Prediction System** is a full-stack machine learning application designed to help HR departments identify employees at risk of leaving the company. By analyzing various factors such as job satisfaction, salary, performance ratings, and workload, the system provides actionable insights and risk scores, enabling proactive retention strategies.

This project features a modular backend for data simulation, preprocessing, and model training, coupled with a modern, interactive frontend for data visualization and risk analysis.

## Key Features

- **Synthetic Data Generation**: Simulates a realistic workforce dataset with customizable parameters.
- **Automated ML Pipeline**: Includes data preprocessing, feature engineering, and training of multiple classification models (Logistic Regression, Random Forest, Gradient Boosting).
- **Risk Scoring Engine**: Calculates real-time risk scores and categorizes employees into High, Medium, and Low risk levels.
- **RESTful API**: A Flask-based API that exposes endpoints for training, prediction, and analytics.
- **Interactive Dashboard**: A clean, responsive UI to visualize attrition rates, department summaries, and individual employee risk profiles.

## Project Structure

```text
attrition-system/
├── backend/
│   ├── app.py              # Flask API Entry Point
│   ├── data/               # Generated Datasets
│   ├── models/             # Trained Models & Preprocessors
│   ├── modules/            # Core Logic (Data Gen, Preprocessing, Training, Scoring)
│   └── requirements.txt    # Python Dependencies
├── frontend/
│   └── index.html          # Interactive Dashboard
└── README.md               # Project Documentation
```

## Setup & Usage

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate data and train models:
   ```bash
   python modules/model_trainer.py
   ```
4. Start the API server:
   ```bash
   python app.py
   ```

### Frontend Usage

Simply open `frontend/index.html` in any modern web browser. The dashboard will automatically connect to the backend API (running on `http://localhost:5000`) to fetch and display analytics.

## Technology Stack

- **Backend**: Python, Flask, Scikit-learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, Chart.js (if applicable or pure CSS/SVG)
- **Deployment**: Git/GitHub for version control
