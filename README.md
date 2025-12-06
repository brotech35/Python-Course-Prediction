# Student Performance Prediction Project

### Deployed App Link
(https://python-course-pred.streamlit.app/)
***
## Student Performance Prediction – End-to-End Data Science Project

### 1. Overview
- This project focuses on building a machine learning system to predict student outcomes (**Pass/Fail**) in an online programming course.
- The workflow includes data cleaning, feature engineering (creating `consistency` and `engagement` metrics), model training, evaluation, and deployment using Streamlit.
- The final model is **Logistic Regression**, selected based on achieving the highest ROC-AUC score.

### 2. Dataset
The dataset contains student information (approx. 3,000 records), including:
- **Demographics:** `age`, `country`.
- **Background:** `prior_programming_experience`.
- **Study Habits:** `hours_spent_learning_per_week`, `weeks_in_course`.
- **Engagement:** `projects_completed`, `debugging_sessions_per_week`, `tutorial_videos_watched`.
- **Target variable:** `passed_exam`.

***
### 3. Project Workflow & Results

#### 3.1 Data Cleaning & Feature Engineering
- Handled missing values (e.g., imputed missing `prior_programming_experience` as "No").
- Applied **One-Hot Encoding** and **Standard Scaling**.
- **Engineered Key Features:**
    - **`consistency`**: Weekly effort.
    - **`engagement`**: Sum of practical activities (problems, projects, debugging sessions).

#### 3.2 Feature Selection
- Used **RandomForestClassifier** feature importance to select the **top 15** most influential predictors, ensuring the final model is focused and efficient.

#### 3.3 Model Training and Selection
- Evaluated multiple models (Logistic Regression, Random Forest, XGBoost).
- **Metric:** **ROC-AUC** was the preferred metric due to the class imbalance.
- **Logistic Regression** delivered the superior performance for this problem.

#### 3.4 Hyperparameter Tuning
- Performed **GridSearchCV** with 5-fold cross-validation on the Logistic Regression model.
- **Best Parameters Found:**
    - `C = 1`
    - `solver = "saga"`
    - `penalty = "l2"`
    - `max_iter = 1000`

- **Final Model Performance:**
    - Accuracy: **~92.5%**
    - ROC-AUC: **~0.95**

These results indicate a highly effective model for identifying student risk.

***

### 4. Model Deployment (Streamlit)
- The final scaled model and scaler were saved using `pickle` and `joblib`.
- **`app.py`** provides an interactive dashboard where users can input student metrics and instantly receive a prediction (Pass/Fail) with probability scores.
- 
### 5. Files in This Project
- `Python Course.ipynb` – Full EDA, preprocessing, feature engineering, and model building
- `app.py` – Streamlit prediction interface.
- `logreg_python_model.pkl` – Trained Logistic Regression model.
- `scaler.pkl` – Trained StandardScaler used for numeric feature scaling.
- `requirements.txt` – Dependencies required for running the application.

### 6. How to Run the Streamlit App

1.  **File Setup:** Ensure `app.py`, `logreg_python_model.pkl`, and `scaler.pkl` are in the same folder.
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute the app:**
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your default browser.
