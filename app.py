import streamlit as st
import pandas as pd
import pickle
import joblib

with open('logreg_python_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
scaler = joblib.load('scaler.pkl')

top_features = [
    'prior_programming_experience', 
    'hours_spent_learning_per_week', 
    'consistency', 
    'projects_completed', 
    'self_reported_confidence_python', 
    'experienced', 
    'engagement', 
    'age', 
    'practice_problems_solved', 
    'tutorial_videos_watched', 
    'debugging_sessions_per_week', 
    'weeks_in_course', 
    'uses_kaggle', 
    'participates_in_discussion_forums', 
    'country_Brazil'
]

NUMERIC_FEATURES = [
    'age', 'weeks_in_course', 'hours_spent_learning_per_week', 
    'practice_problems_solved', 'projects_completed', "tutorial_videos_watched",'debugging_sessions_per_week', 
    'self_reported_confidence_python', 'engagement','consistency',
    'prior_programming_experience'
]

st.title("🎓 Student Performance Prediction")
st.write("Enter student metrics to predict their final exam outcome (Pass/Fail).")

def user_input_features():
    
    st.sidebar.header("Student Profile Input")
    
    age = st.sidebar.slider("Age", 18, 50, 25)
    country = st.sidebar.selectbox("Country", ["Brazil", "USA", "India", "Other"])
    prior_experience_raw = st.sidebar.selectbox("Prior Programming Experience", ["No", "Beginner", "Intermediate", "Advanced"])
    
    # INTEGER inputs now use step=1
    weeks_in_course = st.sidebar.number_input("Weeks in Course", min_value=1, max_value=52, value=12, step=1)
    hours_weekly = st.sidebar.number_input("Avg. Hours Spent Learning Per Week", min_value=1, max_value=30, value=8, step=1)
    problems_solved = st.sidebar.number_input("Practice Problems Solved (Total)", min_value=0, max_value=500, value=50, step=1)
    projects_completed = st.sidebar.number_input("Projects Completed (Total)", min_value=0, max_value=20, value=3, step=1)
    tutorial_videos = st.sidebar.number_input("Tutorial Videos Watched (Total)", min_value=0, max_value=200, value=15, step=1)
    
    # FLOAT input now uses float values and step
    debugging_sessions = st.sidebar.number_input("Debugging Sessions Per Week", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    confidence = st.sidebar.slider("Self-Reported Python Confidence (1=Low, 10=High)", 1, 10, 5)
    uses_kaggle = st.sidebar.selectbox("Uses Kaggle?", ["Yes", "No"])
    participates_forums = st.sidebar.selectbox("Participates in Discussion Forums?", ["Yes", "No"])
    
    consistency = hours_weekly / (weeks_in_course if weeks_in_course > 0 else 1)
    
    engagement = problems_solved + projects_completed + debugging_sessions
    
    experienced = 1 if prior_experience_raw in ['Intermediate', 'Advanced'] else 0

    features = {
        'age': age,
        'weeks_in_course': weeks_in_course,
        'hours_spent_learning_per_week': hours_weekly,
        'practice_problems_solved': problems_solved,
        'projects_completed': projects_completed,
        'debugging_sessions_per_week': debugging_sessions,
        'tutorial_videos_watched': tutorial_videos,
        'self_reported_confidence_python': confidence,
        
        'consistency': consistency,
        'engagement': engagement,
        'experienced': experienced,
        
        'prior_programming_experience': {"No": 0, "Beginner": 1, "Intermediate": 2, "Advanced": 3}.get(prior_experience_raw, 0),
        'country_Brazil': 1 if country == 'Brazil' else 0,
        'uses_kaggle': 1 if uses_kaggle == 'Yes' else 0,
        'participates_in_discussion_forums': 1 if participates_forums == 'Yes' else 0,
    }

    final_input_df = pd.DataFrame(0, index=[0], columns=top_features)
    for col in final_input_df.columns:
        if col in features:
            final_input_df[col] = features[col]

    return final_input_df

input_df = user_input_features()

if st.sidebar.button("Predict Outcome"):
    
    df_scaled = input_df.copy()
    
    cols_to_scale = [col for col in NUMERIC_FEATURES if col in top_features]
    df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])

    prediction = model.predict(df_scaled)[0]
    prediction_prob = model.predict_proba(df_scaled)[0]

    prob_pass = prediction_prob[1] * 100
    prob_fail = prediction_prob[0] * 100

    st.markdown("---")
    
    if prediction == 1:
        st.success(f"## Prediction: Student is likely to PASS! 🎉")
    else:
        st.error(f"## Prediction: Student is likely to FAIL. ⚠️")

    st.metric(label="Chance of Passing", value=f"{prob_pass:.1f}%")
    st.metric(label="Chance of Failing", value=f"{prob_fail:.1f}%")
    