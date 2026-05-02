
# Import necessary libraries
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ==========================================
# DOWNLOAD AND LOAD THE TRAINED MODEL
# ==========================================
# Download the trained model from Hugging Face model hub
# Use the repo ID for your uploaded model (update the <---repo id----> placeholder)
model_path = hf_hub_download(repo_id="sandeep8617/visit-with-us-tourism-project", filename="wellness_tourism_model_v1.joblib")
model = joblib.load(model_path)

# ==========================================
# SET UP THE STREAMLIT USER INTERFACE (UI)
# ==========================================
# Display the title and a brief description for the user
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts the likelihood of a customer purchasing the **Wellness Tourism Package**.
Based on the customer's characteristics such as age, city tier, gender, etc., we will estimate if they are a potential buyer.
Please enter the customer details below to get the purchase likelihood prediction.
""")

# ==========================================
# USER INPUT FIELDS FOR CUSTOMER DETAILS
# ==========================================
# Get the user input for customer attributes
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# City tier input (handled as integer values)
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
if city_tier == "Tier 1":
    city_tier_value = 1
elif city_tier == "Tier 2":
    city_tier_value = 2
else:
    city_tier_value = 3

occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Additional customer-related features
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
monthly_income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=100000, value=5000, step=100)
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=100, value=5, step=1)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=50, value=3, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (Minutes)", min_value=1, max_value=100, value=15, step=1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
passport = st.selectbox("Has Passport?", ["Yes", "No"])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Features related to customer interaction
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)
own_car = st.selectbox("Own Car?", ["Yes", "No"])

# ==========================================
# ASSEMBLE USER INPUT INTO A DATAFRAME
# ==========================================
# Create a DataFrame from user inputs for prediction
input_data = pd.DataFrame([{
    'age': age,
    'city_tier': city_tier_value,  # Use the mapped integer value for city tier
    'occupation': occupation,
    'gender': gender,
    'marital_status': marital_status,
    'monthly_income': monthly_income,
    'number_of_trips': number_of_trips,
    'number_of_followups': number_of_followups,
    'duration_of_pitch': duration_of_pitch,
    'product_pitched': product_pitched,
    'preferred_property_star': preferred_property_star,
    'passport': passport,
    'pitch_satisfaction_score': pitch_satisfaction_score,
    'number_of_person_visiting': number_of_person_visiting,
    'number_of_children_visiting': number_of_children_visiting,
    'own_car': own_car
}])

# ==========================================
# CLASSIFICATION THRESHOLD
# ==========================================
# Set the classification threshold (e.g., 0.50)
classification_threshold = 0.50

# ==========================================
# PREDICTION BUTTON FOR CUSTOMER PURCHASE LIKELIHOOD
# ==========================================
# When the user presses the "Predict Purchase Likelihood" button, make a prediction
if st.button("Predict Purchase Likelihood"):
    # Get prediction probability from the model (predict_proba method)
    prediction_proba = model.predict_proba(input_data)[0, 1]
    
    # Predict based on classification threshold
    prediction = (prediction_proba >= classification_threshold).astype(int)
    
    # Convert the prediction to a readable result
    result = "Potential Customer" if prediction == 1 else "Not a Customer"
    
    # Display the result
    st.subheader("Prediction Result:")  # Heading for result
    st.success(f"Based on the information provided, the customer is likely to: **{result}**")
    st.write(f"Prediction Probability: **{prediction_proba*100:.2f}%**")  # Optionally display the probability
