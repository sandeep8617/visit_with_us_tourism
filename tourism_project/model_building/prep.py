
# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

# Import pandas for data manipulation and analysis
import pandas as pd

# Import os module for accessing environment variables
import os

# Import train_test_split for splitting dataset into train and test sets
from sklearn.model_selection import train_test_split

# Import LabelEncoder for converting categorical values into numeric values
from sklearn.preprocessing import LabelEncoder

# Import Hugging Face API for uploading processed datasets
from huggingface_hub import HfApi


# ==========================================
# CONNECT TO HUGGING FACE HUB
# ==========================================

# Define Hugging Face dataset repository name
repo_id = "sandeep8617/visit-with-us-tourism-project"

# Initialize Hugging Face API using token stored in environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))


# ==========================================
# LOAD DATASET
# ==========================================

# Define dataset path stored on Hugging Face Hub
DATASET_PATH = "hf://datasets/sandeep8617/visit-with-us-tourism-project/tourism.csv"

# Load dataset into pandas DataFrame
df = pd.read_csv(DATASET_PATH)

# Print confirmation message after loading dataset
print("Dataset loaded successfully.")


# ==========================================
# DATA CLEANING
# ==========================================

# Drop unnecessary identifier columns
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)


# ==========================================
# HANDLE INCONSISTENT CATEGORICAL VALUES
# ==========================================

# Replace incorrect gender value "Fe Male" with "Female"
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Replace "Unmarried" with "Single" for standardization
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')


# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

# Create list of all categorical columns
categorical_columns = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Loop through each categorical column
for col in categorical_columns:

    # Create new LabelEncoder object for each column
    encoder = LabelEncoder()

    # Convert categorical values into numerical values
    df[col] = encoder.fit_transform(df[col])


# ==========================================
# DEFINE TARGET VARIABLE
# ==========================================

# Define target column for prediction
target_col = 'ProdTaken'


# ==========================================
# SPLIT FEATURES AND TARGET
# ==========================================

# Create feature dataset by removing target column
X = df.drop(columns=[target_col])

# Create target dataset
y = df[target_col]


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

# Split dataset into training and testing sets
# test_size=0.2 means 20% data used for testing
# random_state ensures reproducibility
# stratify maintains target class distribution
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Print confirmation after train-test split
print("Train-test split completed successfully.")


# ==========================================
# SAVE PROCESSED DATASETS
# ==========================================

# Save training features dataset
Xtrain.to_csv("Xtrain.csv", index=False)

# Save testing features dataset
Xtest.to_csv("Xtest.csv", index=False)

# Save training target dataset
ytrain.to_csv("ytrain.csv", index=False)

# Save testing target dataset
ytest.to_csv("ytest.csv", index=False)

# Print confirmation after saving files
print("Processed datasets saved successfully.")


# ==========================================
# UPLOAD DATA TO HUGGING FACE HUB
# ==========================================

# Create list of files to upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Loop through each file
for file_path in files:

    # Upload each file to Hugging Face dataset repository
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
    )

# Print final success message
print("All files uploaded successfully to Hugging Face Hub.")
