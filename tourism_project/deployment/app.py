
# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

# Import Streamlit for web app UI
import streamlit as st

# Import pandas for dataframe creation
import pandas as pd

# Import Hugging Face utility to download model
from huggingface_hub import hf_hub_download

# Import joblib to load trained model
import joblib


# ==========================================
# DOWNLOAD AND LOAD TRAINED MODEL
# ==========================================

# Download trained model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="sandeep8617/visit-with-us-tourism-project-model",
    filename="best_tourism_package_model_v1.joblib"
)

# Load trained model
model = joblib.load(model_path)


# ==========================================
# STREAMLIT APPLICATION TITLE
# ==========================================
# Display the title and a brief description for the user
st.title("Wellness Tourism Package Purchase Prediction")

st.write("""
This application predicts whether a customer is likely to purchase the newly launched
**Wellness Tourism Package** based on customer demographics and interaction details.
""")


# ==========================================
# USER INPUT SECTION
# ==========================================
# Get the user input for customer attributes
# Customer age
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# Type of contact
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])

# City tier input (handled as integer values)
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
if city_tier == "Tier 1":
    city_tier_value = 1
elif city_tier == "Tier 2":
    city_tier_value = 2
else:
    city_tier_value = 3

# Occupation
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])

# Number of persons visiting
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)

# Preferred property star
preferred_property_star = st.selectbox("Preferred Property Star",[1, 2, 3, 4, 5])

# Marital status
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

# Number of trips
number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=30, value=5, step=1)

# Passport
passport = st.selectbox("Has Passport?", ["Yes", "No"])

# Own car
own_car = st.selectbox("Own Car", ["Yes", "No"])

# Number of children visiting
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)

# Designation
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Monthly income
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000, step=100)

# Pitch satisfaction score
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Product pitched
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# Number of followups
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=50, value=3, step=1)

# Duration of pitch
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1, max_value=200, value=15, step=1)

# ==========================================
# ASSEMBLE USER INPUT INTO A DATAFRAME
# ==========================================
# Create a DataFrame from user inputs for prediction
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier_value,  # Use the mapped integer value for city tier
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
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
    result = "Likely to Purchase a Package" if prediction == 1 else "Less Likely to Purchase a Package"

    # Display the result
    st.subheader("Prediction Result:")  # Heading for result
    st.success(f"Based on the information provided, the customer is likely to: **{result}**")
    st.write(f"Prediction Probability: **{prediction_proba*100:.2f}%**")  # Optionally display the probability
