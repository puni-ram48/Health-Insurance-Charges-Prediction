import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import base64

# Load the model
model = pickle.load(open('svm_model.pkl', 'rb'))

# Initialize the scaler
scaler = MinMaxScaler()

# Function to add a background image and custom CSS
def add_bg_and_custom_css(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Comic+Sans+MS&display=swap');
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            font-family: 'Comic Sans MS', cursive, sans-serif; /* Apply Comic Sans font */
        }}
        .block-container {{
            width: 100%;
            padding-top: 10px; /* Remove top padding */
            border-radius: 10px; /* Rounded corners for the container */
            padding: 20px; /* Padding inside the container */
            margin-top: 10px; /* Remove top margin */
        }}
        .input-label {{
            font-weight: bold;
            font-size: 20px;
            color: #333;
            margin-bottom: 10px;
            margin-top: 0px; /* Remove top margin */
            font-family: 'Comic Sans MS', cursive, sans-serif; /* Apply Comic Sans font */
        }}
        .predicted-charges {{
            font-weight: bold;
            font-size: 24px;
            color: black;
            font-family: 'Comic Sans MS', cursive, sans-serif; /* Apply Comic Sans font */
        }}
        .title {{
            margin-bottom: 0px; /* Remove bottom margin for title */
            font-family: 'Comic Sans MS', cursive, sans-serif; /* Apply Comic Sans font */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background image and custom CSS
add_bg_and_custom_css('image.png')

# Streamlit app
st.markdown('<h1 class="title">Insurance Charges Prediction</h1>', unsafe_allow_html=True)

# Collect input data with custom label styling
def custom_input_label(label):
    st.markdown(f'<p class="input-label">{label}</p>', unsafe_allow_html=True)

# Start block container
st.markdown('<div class="block-container">', unsafe_allow_html=True)

custom_input_label('Age')
age = st.number_input('Age', min_value=0, max_value=100, value=25, label_visibility='collapsed')

custom_input_label('Gender')
gender = st.selectbox('Gender', ('male', 'female'), label_visibility='collapsed')

custom_input_label('BMI')
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0, label_visibility='collapsed')

custom_input_label('Smoker')
smoker = st.selectbox('Smoker', ('yes', 'no'), label_visibility='collapsed')

custom_input_label('Number of Children')
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0, label_visibility='collapsed')

custom_input_label('Region')
region = st.selectbox('Region', ('southwest', 'northwest', 'northeast', 'southeast'), label_visibility='collapsed')

# Convert inputs to the format expected by the model
gender = 1 if gender == 'male' else 0
smoker = 1 if smoker == 'yes' else 0
region_dict = {'southwest': 0, 'northwest': 1, 'northeast': 2, 'southeast': 3}
region = region_dict[region]

# Prepare the input features as a DataFrame with correct column names
input_features = pd.DataFrame({
    'age': [age],
    'sex_male': [gender],
    'bmi': [bmi],
    'smoker': [smoker],
    'children': [children],
    'region': [region]
})

# Scale the numerical features
input_features[['age', 'bmi']] = scaler.fit_transform(input_features[['age', 'bmi']])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_features)
    output = round(np.exp(prediction[0]), 2)
    st.markdown(f'<p class="predicted-charges">Predicted Charges: ${output}</p>', unsafe_allow_html=True)

# Close the block container
st.markdown('</div>', unsafe_allow_html=True) 