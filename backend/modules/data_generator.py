"""
Module 1: Data Generation
Generates a simulated employee workforce dataset.
"""

import pandas as pd
import numpy as np
import os


def generate_dataset(n_employees: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    departments = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
    job_roles = {
        "Engineering": ["Software Engineer", "Data Analyst", "DevOps Engineer"],
        "Sales": ["Sales Representative", "Account Manager", "Sales Lead"],
        "HR": ["HR Specialist", "Recruiter", "HR Manager"],
        "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
        "Marketing": ["Marketing Analyst", "Content Strategist", "Brand Manager"],
        "Operations": ["Operations Analyst", "Logistics Manager", "Supply Chain Specialist"],
    }

    records = []
    for emp_id in range(1, n_employees + 1):
        dept = np.random.choice(departments)
        role = np.random.choice(job_roles[dept])
        age = int(np.random.randint(22, 60))
        years_at_company = int(np.random.randint(0, min(age - 21, 30)))
        salary = int(np.random.normal(60000, 20000))
        salary = max(25000, min(salary, 150000))
        performance = round(np.random.uniform(1.0, 5.0), 1)
        satisfaction = round(np.random.uniform(1.0, 5.0), 1)
        work_hours = int(np.random.normal(45, 8))
        work_hours = max(30, min(work_hours, 80))
        promotions = int(np.random.poisson(years_at_company / 3)) if years_at_company > 0 else 0

        # Attrition logic (realistic bias)
        attrition_score = 0
        if satisfaction < 2.5:
            attrition_score += 3
        if salary < 40000:
            attrition_score += 2
        if work_hours > 60:
            attrition_score += 2
        if performance < 2.5:
            attrition_score += 1
        if promotions == 0 and years_at_company > 3:
            attrition_score += 2
        if years_at_company < 2:
            attrition_score += 1

        attrition_prob = attrition_score / 11
        attrition = "Yes" if np.random.random() < attrition_prob else "No"

        records.append({
            "EmployeeID": f"EMP{emp_id:04d}",
            "Age": age,
            "Department": dept,
            "JobRole": role,
            "Salary": salary,
            "YearsAtCompany": years_at_company,
            "PerformanceRating": performance,
            "JobSatisfaction": satisfaction,
            "WorkHours": work_hours,
            "Promotions": promotions,
            "Attrition": attrition,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"\nAttrition Rate: {(df['Attrition'] == 'Yes').mean():.2%}")
    
    # Path relative to backend/data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "employee_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")
