import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title='Calories Prediction', layout='centered')
st.title('Calories Burned Prediction')
st.caption('Model: XGBoost | Encoding: Male=0, Female=1')

# Load model and scaler
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = len(scaler.mean_)

# --- STEP 1: TWO FILE UPLOAD FIELDS STACKED ---
st.header('1. Upload Your Datasets')
exercise_file = st.file_uploader("Upload exercise.csv", type=['csv'], key='ex')
calories_file = st.file_uploader("Upload calories.csv", type=['csv'], key='cal')

# --- STEP 2: R² AND MAE APPEAR HERE AFTER BOTH UPLOADS ---
if exercise_file is not None and calories_file is not None:
    df_exercise = pd.read_csv(exercise_file)
    df_calories = pd.read_csv(calories_file)
    
    # Merge datasets
    if 'User_ID' in df_exercise.columns and 'User_ID' in df_calories.columns:
        df = pd.merge(df_exercise, df_calories, on='User_ID')
    else:
        df = pd.concat([df_exercise, df_calories], axis=1)
    
    # Prepare X and y - adjust column names if yours are different
    X = df[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
    y_true = df['Calories']
    
    # CHANGED: Male=0, Female=1
    X['Gender'] = X['Gender'].astype(str).str.strip().str.capitalize().map({'Male': 0, 'Female': 1})
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    test_r2 = r2_score(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    
    st.subheader('2. Model Performance on Uploaded Data')
    st.success(f"**Test R²: {test_r2:.4f} | Test MAE: {test_mae:.2f} kcal**")
    
    with st.expander("See merged data preview"):
        st.dataframe(df.head())
    
    st.divider()
    
    # --- STEP 3: INPUT FIELDS APPEAR UNDER R²/MAE ---
    st.subheader('3. Predict for a Single Entry')
    
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', 10, 100, 25)
    height = st.number_input('Height (cm)', 100, 250, 170)
    weight = st.number_input('Weight (kg)', 30, 200, 70)
    duration = st.number_input('Duration (min)', 1, 300, 30)
    heart_rate = st.number_input('Heart Rate (bpm)', 40, 200, 120)
    body_temp = st.number_input('Body Temp (°C)', 35.0, 45.0, 37.0, 0.1)
    
    if st.button('Predict Calories'):
        # CHANGED: Male=0, Female=1
        gender_val = 0 if gender == 'Male' else 1
        input_data = pd.DataFrame([[gender_val, age, height, weight, duration, heart_rate, body_temp]],
                                  columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.success(f'**Predicted Calories Burned: {prediction:.2f} kcal**')

else:
    st.info("Upload exercise.csv first, then calories.csv. R²/MAE and input fields will appear after both are uploaded.")