"""
Module 4: Risk Scoring Engine
Generates individual and batch employee risk scores with HR insights.
"""

import pandas as pd
import numpy as np


class RiskScoringEngine:
    def __init__(self, preprocessor, trainer):
        self.preprocessor = preprocessor
        self.trainer = trainer

    def score_employee(self, employee_dict: dict) -> dict:
        X = self.preprocessor.transform_single(employee_dict)
        result = self.trainer.predict_risk(X)

        # HR Insights
        insights = self._generate_insights(employee_dict, result["risk_score"])
        result["insights"] = insights
        result["employee_id"] = employee_dict.get("EmployeeID", "N/A")
        return result

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            emp = row.to_dict()
            score = self.trainer.predict_risk(
                self.preprocessor.transform_single(emp)
            )
            results.append({
                "EmployeeID": emp.get("EmployeeID", "N/A"),
                "Department": emp.get("Department", ""),
                "JobRole": emp.get("JobRole", ""),
                "RiskScore": score["risk_score"],
                "RiskLevel": score["risk_level"],
                "AttritionProbability": score["attrition_probability"],
                "Prediction": score["prediction"],
            })
        return pd.DataFrame(results).sort_values("RiskScore", ascending=False)

    def _generate_insights(self, emp: dict, risk_score: float) -> list:
        insights = []

        if emp.get("JobSatisfaction", 5) < 2.5:
            insights.append("⚠️ Low job satisfaction — consider a 1:1 review or role adjustment.")
        if emp.get("Salary", 999999) < 40000:
            insights.append("💰 Below-market salary — benchmark against industry standards.")
        if emp.get("WorkHours", 40) > 60:
            insights.append("⏰ Excessive work hours — risk of burnout, review workload distribution.")
        if emp.get("Promotions", 0) == 0 and emp.get("YearsAtCompany", 0) > 3:
            insights.append("📈 No promotions in 3+ years — discuss career growth opportunities.")
        if emp.get("PerformanceRating", 5) < 2.5:
            insights.append("📉 Low performance rating — consider mentoring or support programs.")
        if emp.get("YearsAtCompany", 99) < 2:
            insights.append("🆕 Early tenure employee — strengthen onboarding and engagement.")

        if not insights:
            if risk_score < 33:
                insights.append("✅ Employee appears engaged and stable. Maintain regular check-ins.")
            elif risk_score < 66:
                insights.append("🔍 Monitor closely — some risk factors present but manageable.")
            else:
                insights.append("🚨 High attrition risk — immediate intervention recommended.")

        return insights

    def get_department_summary(self, scored_df: pd.DataFrame) -> list:
        summary = scored_df.groupby("Department").agg(
            AvgRiskScore=("RiskScore", "mean"),
            HighRiskCount=("RiskLevel", lambda x: (x == "High").sum()),
            TotalEmployees=("EmployeeID", "count"),
        ).reset_index()
        summary["AvgRiskScore"] = summary["AvgRiskScore"].round(1)
        return summary.to_dict(orient="records")


if __name__ == "__main__":
    print("Risk Scoring Engine module loaded successfully.")
