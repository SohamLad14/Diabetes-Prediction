import streamlit as st
import pickle
import pandas as pd

# Set the background image and text color
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.indianexpress.com/2023/11/diabetes-2.jpg?w=640");
    background-size: 100vw 100vh;
    background-position: center;  
    background-repeat: no-repeat;
    color: #FFFFFF; /* Set text color to blue */
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)
# Load the trained model
with open("diabetes_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Perform any necessary preprocessing steps here (e.g., scaling)
    scaled_data = scaler.transform(data)
    return scaled_data

# Function to predict diabetes
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age]
    })
    input_data_scaled = preprocess_input(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit app
def main():
    st.title("Diabetes Prediction")

    # Input fields for user to enter data
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=846, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=2.42, step=0.01)
    age = st.number_input("Age", min_value=21, max_value=81, step=1)

    # Button to trigger prediction
    if st.button("Predict"):
        prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        if prediction[0] == 0:
            st.success("No Diabetes Detected")
        else:
            st.error("Diabetes Detected")

if __name__ == "__main__":
    main()
