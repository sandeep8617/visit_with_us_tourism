
# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

import pandas as pd

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Encoding categorical variables into numeric format
from sklearn.preprocessing import LabelEncoder

# Hugging Face Hub for dataset storage in MLOps pipeline
from huggingface_hub import HfApi
import os


# ==========================================
# CONNECT TO HUGGING FACE HUB
# ==========================================

# Initialize Hugging Face API using secure token from environment variables
# This enables dataset upload as part of MLOps CI/CD pipeline
api = HfApi(token=os.getenv("HF_TOKEN"))


# ==========================================
# LOAD DATASET
# ==========================================

# Load cleaned dataset for Wellness Tourism Package prediction
# This dataset contains customer demographics and interaction details
# Path to the dataset stored on Hugging Face Hub under the "visit-with-us-tourism-project" space
DATASET_PATH = "hf://datasets/sandeep8617/visit-with-us-tourism-project/tourism.csv"

# Load the data into pandas DataFrame
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# ==========================================
# DATA CLEANING (FEATURE ENGINEERING STEP 1)
# ==========================================

# Drop identifier columns as they do not contribute to prediction
# These columns have unique values and do not provide learning patterns
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)


# ==========================================
# HANDLE INCONSISTENT CATEGORICAL VALUES
# ==========================================

# Standardizing inconsistent labels in Gender column
# "Fe Male" is corrected to "Female" for consistency
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Standardizing MaritalStatus values
# "Unmarried" is merged into "Single" to avoid duplicate categories
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')


# ==========================================
# ENCODING CATEGORICAL VARIABLES
# ==========================================

# Label Encoding converts categorical variables into numerical format
# Required for machine learning model compatibility

label_encoder = LabelEncoder()

df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])


# ==========================================
# DEFINE TARGET VARIABLE
# ==========================================

# Target variable:
# 1 → Customer purchased Wellness Tourism Package
# 0 → Customer did not purchase package
target_col = 'ProdTaken'


# ==========================================
# SPLIT FEATURES AND TARGET
# ==========================================

# Separating independent features (X) and target variable (y)
X = df.drop(columns=[target_col])
y = df[target_col]


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

# Splitting dataset into training and testing sets
# 80% training data, 20% testing data
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# SAVE PROCESSED DATASETS
# ==========================================

# Saving datasets for reproducibility in MLOps pipeline
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test split completed successfully.")


# ==========================================
# UPLOAD DATA TO HUGGING FACE HUB
# ==========================================

# Upload processed datasets to Hugging Face Hub for version control
# These files are uploaded to the "visit-with-us-tourism-project" space for dataset tracking
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1], # just the filename
        repo_id="sandeep8617/visit-with-us-tourism-project",  # Correct Hugging Face space name
        repo_type="dataset",
    )

print("All files uploaded successfully to Hugging Face Hub.")
