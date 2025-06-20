# INX Future Inc. Employee Performance Prediction Model

This project aims to predict employee performance ratings for INX Future Inc. using a Random Forest Classifier. It integrates data preprocessing, exploratory analysis, machine learning, and interactive visualizations to support performance management and HR decision-making.

---

## Project Summary

- **Objective:** Predict employee performance based on demographic, job-related, and satisfaction features.
- **Algorithm:** Random Forest Classifier
- **Tools Used:**  
  - Python (Pandas, Scikit-learn, Matplotlib, Seaborn)  
  - GridSearchCV (for hyperparameter tuning)  
  - LabelEncoder (for categorical data)  
  - Power BI (for dashboard visualizations)  
  - Streamlit (for interactive model deployment)  
  - Google Colab (for development and experimentation)  

---

## Key Insights

- **Marital Status:** Divorced employees exhibit significantly lower performance, possibly due to stress or work-life imbalance.  
- **Age:** Peak performance is found among employees aged 26–45, suggesting experience and maturity are important.  
- **Job Satisfaction & Environment:** Higher job satisfaction, relationship satisfaction, and environment satisfaction all positively correlate with better performance.  
- **Attrition:** Lower attrition is associated with higher performance, indicating that stable teams perform better.  
- **Overtime:** Employees who do **not** work overtime perform better, suggesting overtime could lead to burnout or lower productivity.  
- **Work-Life Balance:** A moderately high work-life balance is strongly linked to high performance.  
- **Distance from Home:** Employees who live closer to work tend to perform better, highlighting the impact of commuting.  
- **Business Travel:** Frequent travelers perform worse than those who travel rarely.  
- **Promotion & Manager Experience:** Employees with more recent promotions and consistent managers tend to perform better.  
- **Job Involvement:** Moderate to high involvement correlates with stronger performance.  
- **Education & Department Alignment:** Employees in life sciences and medical fields, working in aligned departments, show higher performance.  

---

## Observations Without Strong Correlation

- **Hourly Rate & Training Frequency:** No strong impact on performance.  
- **Years of Total Experience:** Slight negative correlation with performance.  
- **Salary Hike Percent:** Unexpectedly, higher hikes were associated with lower performance, suggesting other dynamics may be in play.  
- **Tenure in Same Role/Company:** Longer tenure negatively correlated with performance, possibly due to stagnation.  

---

## Tools & Techniques

- **Label Encoding:** Converted categorical variables to numeric format  
- **Random Forest Classifier:** Chosen for robustness with mixed feature types  
- **GridSearchCV:** Tuned model hyperparameters using cross-validation  
- **EDA (Exploratory Data Analysis):** Identified key performance drivers  
- **Streamlit:** Built an interactive interface for model use and evaluation  
- **Power BI:** Created professional visuals for performance breakdown  

---

## Files Included

- `employee_performance.ipynb` – Full analysis and model pipeline  
- `model_script.py` – Python script version (optional)  
- `report.pdf` – Executive summary with insights and visualizations  
- `dashboard.pbix` – Power BI dashboard file  
- `employee_data.csv` – Raw dataset  

---

## Setup Instructions

Before running the notebook or script, install the following packages:

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
````

If you're using Jupyter or Colab:

```bash
pip install notebook
```

---

## How to Run

To launch the Streamlit app:

```bash
streamlit run model_script.py
```

Or simply open the notebook and run it step-by-step to explore the data and build the model.

---

## Future Enhancements

* Add model comparison (e.g., XGBoost, SVM)
* Integrate SHAP for model explainability
* Deploy model with Streamlit Cloud or Hugging Face Spaces
* Connect Power BI to a live backend for real-time performance monitoring

---

## Author

**Cindy Njeri Kiarie**
*Data Analyst · MSc Data Science*
