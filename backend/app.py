"""
Backend API: Flask REST API
Exposes all modules via HTTP endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
import pandas as pd
import json

from modules.data_generator import generate_dataset
from modules.preprocessor import DataPreprocessor
from modules.model_trainer import ModelTrainer
from modules.risk_scoring import RiskScoringEngine

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
preprocessor: DataPreprocessor = None
trainer: ModelTrainer = None
risk_engine: RiskScoringEngine = None
dataset: pd.DataFrame = None
scored_dataset: pd.DataFrame = None


def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

app.after_request(add_cors_headers)


@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        from flask import Response
        return Response(status=200)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "models_trained": trainer is not None,
        "dataset_loaded": dataset is not None,
    })


@app.route("/api/train", methods=["POST"])
def train():
    global preprocessor, trainer, risk_engine, dataset, scored_dataset

    n = request.json.get("n_employees", 1000) if request.json else 1000

    # Generate data
    dataset = generate_dataset(n_employees=n)

    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    trainer = ModelTrainer()
    results = trainer.train_all(X_train, X_test, y_train, y_test)

    # Feature importance
    fi = trainer.get_feature_importance(preprocessor.feature_columns)

    # Risk engine
    risk_engine = RiskScoringEngine(preprocessor, trainer)
    scored_dataset = risk_engine.score_batch(dataset)

    # Build response (strip roc curve data from summary)
    summary = {}
    for name, r in results.items():
        summary[name] = {k: v for k, v in r.items() if k not in ("roc_fpr", "roc_tpr", "confusion_matrix")}

    return jsonify({
        "message": "Training complete",
        "best_model": trainer.best_model_name,
        "model_results": summary,
        "feature_importance": fi,
        "attrition_rate": round((dataset["Attrition"] == "Yes").mean() * 100, 1),
        "total_employees": len(dataset),
    })


@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    if dataset is None:
        return jsonify({"error": "No dataset. Call /api/train first."}), 400
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    start = (page - 1) * page_size
    end = start + page_size
    records = dataset.iloc[start:end].to_dict(orient="records")
    return jsonify({
        "data": records,
        "total": len(dataset),
        "page": page,
        "page_size": page_size,
    })


@app.route("/api/model/results", methods=["GET"])
def model_results():
    if trainer is None:
        return jsonify({"error": "Models not trained yet."}), 400

    results = {}
    for name, r in trainer.results.items():
        results[name] = {k: v for k, v in r.items() if k not in ("roc_fpr", "roc_tpr")}

    return jsonify({
        "results": results,
        "best_model": trainer.best_model_name,
    })


@app.route("/api/model/roc", methods=["GET"])
def roc_curves():
    if trainer is None:
        return jsonify({"error": "Models not trained."}), 400
    roc_data = {}
    for name, r in trainer.results.items():
        roc_data[name] = {
            "fpr": r["roc_fpr"],
            "tpr": r["roc_tpr"],
            "auc": r["roc_auc"],
        }
    return jsonify(roc_data)


@app.route("/api/model/feature-importance", methods=["GET"])
def feature_importance():
    if trainer is None:
        return jsonify({"error": "Models not trained."}), 400
    fi = trainer.get_feature_importance(preprocessor.feature_columns)
    return jsonify(fi)


@app.route("/api/predict", methods=["POST"])
def predict():
    if risk_engine is None:
        return jsonify({"error": "Models not trained."}), 400

    data = request.json
    required = ["Age", "Department", "JobRole", "Salary", "YearsAtCompany",
                "PerformanceRating", "JobSatisfaction", "WorkHours", "Promotions"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    result = risk_engine.score_employee(data)
    return jsonify(result)


@app.route("/api/risk/scores", methods=["GET"])
def risk_scores():
    if scored_dataset is None:
        return jsonify({"error": "No scored data. Train first."}), 400
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    risk_filter = request.args.get("risk_level", None)

    df = scored_dataset.copy()
    if risk_filter:
        df = df[df["RiskLevel"] == risk_filter]

    total = len(df)
    start = (page - 1) * page_size
    records = df.iloc[start:start + page_size].to_dict(orient="records")
    return jsonify({"data": records, "total": total, "page": page, "page_size": page_size})


@app.route("/api/risk/department-summary", methods=["GET"])
def dept_summary():
    if scored_dataset is None:
        return jsonify({"error": "No data. Train first."}), 400
    summary = risk_engine.get_department_summary(scored_dataset)
    return jsonify(summary)


@app.route("/api/analytics/overview", methods=["GET"])
def analytics_overview():
    if scored_dataset is None:
        return jsonify({"error": "No data."}), 400

    high = int((scored_dataset["RiskLevel"] == "High").sum())
    medium = int((scored_dataset["RiskLevel"] == "Medium").sum())
    low = int((scored_dataset["RiskLevel"] == "Low").sum())
    total = len(scored_dataset)

    dept_counts = dataset.groupby("Department")["Attrition"].apply(
        lambda x: round((x == "Yes").mean() * 100, 1)
    ).to_dict()

    return jsonify({
        "total_employees": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "attrition_rate": round((dataset["Attrition"] == "Yes").mean() * 100, 1),
        "attrition_by_department": dept_counts,
        "avg_risk_score": round(float(scored_dataset["RiskScore"].mean()), 1),
    })


if __name__ == "__main__":
    print("Starting Employee Attrition Risk Prediction API...")
    app.run(debug=True, port=5000)
