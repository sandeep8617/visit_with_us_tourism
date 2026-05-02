
# ==========================================
# IMPORT NECESSARY LIBRARIES
# ==========================================
# For data manipulation and preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# For model training, hyperparameter tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# For model serialization (saving the model)
import joblib

# For environment and Hugging Face API
import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# For experiment tracking with MLflow
import mlflow


# ==========================================
# MLflow CONFIGURATION
# ==========================================
# Set the tracking URI to log experiments in MLflow (local setup)
mlflow.set_tracking_uri("http://localhost:5000")

# Set the name of the experiment in MLflow for tracking purposes
mlflow.set_experiment("tourism-package-purchase-prediction-experiment")


# ==========================================
# HUGGING FACE CONFIGURATION
# ==========================================
# Initialize the Hugging Face API client to upload files and manage datasets
api = HfApi()

# Define paths for training and testing datasets on Hugging Face
Xtrain_path = "hf://datasets/sandeep8617/visit-with-us-tourism-project/Xtrain.csv"
Xtest_path = "hf://datasets/sandeep8617/visit-with-us-tourism-project/Xtest.csv"
ytrain_path = "hf://datasets/sandeep8617/visit-with-us-tourism-project/ytrain.csv"
ytest_path = "hf://datasets/sandeep8617/visit-with-us-tourism-project/ytest.csv"

# Load the train and test datasets from Hugging Face repository
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# ==========================================
# FEATURE SELECTION AND PREPROCESSING
# ==========================================
# Define numeric and categorical features based on the "Tourism" dataset
numeric_features = [
    'Age', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups', 
    'DurationOfPitch', 'PreferredPropertyStar', 'NumberOfPersonVisiting', 
    'NumberOfChildrenVisiting'
]

categorical_features = [
    'CityTier', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 'TypeofContact', 'Designation'
]

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# ==========================================
# DATA PREPROCESSING PIPELINE
# ==========================================


# Create a preprocessor to scale numeric features and encode categorical features using OneHotEncoding
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),  # Apply scaling to numeric features
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Apply one-hot encoding to categorical features
)


# ==========================================
# MODEL INITIALIZATION AND HYPERPARAMETER TUNING
# ==========================================
# Using XGBClassifier for binary classification (1 for purchase, 0 for no purchase)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, n_jobs=-1)

# Define the hyperparameter grid for tuning the XGBoost model using GridSearchCV
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],  # Number of boosting rounds
    'xgbclassifier__max_depth': [3, 5],  # Maximum depth of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05],  # Step size shrinking
    'xgbclassifier__subsample': [0.7, 0.8],  # Fraction of samples used for each tree
    'xgbclassifier__colsample_bytree': [0.7, 0.8],  # Fraction of features used for each tree
    'xgbclassifier__colsample_bylevel': [0.7, 0.8], # Fraction of features used for each level
    'xgbclassifier__reg_lambda': [0.1, 1]  # L2 regularization term
}

# Create a pipeline that first applies preprocessing and then fits the XGBoost model
model_pipeline = make_pipeline(preprocessor, xgb_model)


# ==========================================
# TRAINING THE MODEL AND TRACKING WITH MLflow
# ==========================================
# Start an MLflow run to track this experiment
with mlflow.start_run():
    # Perform Grid Search to identify the best hyperparameters for the model
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')  # Use accuracy for scoring
    grid_search.fit(Xtrain, ytrain)

    # Log each parameter set and its corresponding mean score to MLflow
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]  # Extract the parameters for the current model
        mean_score = results['mean_test_score'][i]  # Extract the mean score for the current model
        std_score = results['std_test_score'][i]  # Extract the standard deviation for the current model

        # Log the parameters and score to MLflow
        with mlflow.start_run(nested=True):  # Start a nested run for each parameter set
            mlflow.log_params(param_set)  # Log the current parameter set
            mlflow.log_metric("mean_accuracy", mean_score)  # Log the mean accuracy score
            mlflow.log_metric("std_test_score", std_score)  # Log the standard deviation of the test score

    # Log the best parameters found by GridSearchCV for the final model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_  # Get the best model after hyperparameter tuning

    # Define classification threshold (0.45 for binary classification)
    classification_threshold = 0.50

    # Make predictions on both the training and testing datasets using the best model
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]  # Probability for the positive class
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)  # Apply threshold

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]  # Probability for the positive class
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)  # Apply threshold

    # Calculate classification metrics (Accuracy, Precision, Recall, F1 Score, AUC-ROC)
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    train_precision = precision_score(ytrain, y_pred_train)
    train_recall = recall_score(ytrain, y_pred_train)
    train_f1 = f1_score(ytrain, y_pred_train)
    train_roc_auc = roc_auc_score(ytrain, y_pred_train)

    test_accuracy = accuracy_score(ytest, y_pred_test)
    test_precision = precision_score(ytest, y_pred_test)
    test_recall = recall_score(ytest, y_pred_test)
    test_f1 = f1_score(ytest, y_pred_test)
    test_roc_auc = roc_auc_score(ytest, y_pred_test)

    # Log all the metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_roc_auc": train_roc_auc,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc
    })

    # Save the best model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)  # Save the best model to a file

    # Log the saved model as an artifact in the MLflow experiment
    mlflow.log_artifact(model_path, artifact_path="model")  # Log the model file as an artifact
    print(f"Model saved as artifact at: {model_path}")

    # Define the Hugging Face repo ID where the model will be uploaded
    repo_id = "sandeep8617/visit-with-us-tourism-project-model"
    repo_type = "model"  # Define the type of space as 'model'

    # Check if the Hugging Face repository exists, and create it if not
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)  # Check if the space exists
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)  # Create the repo if not found
        print(f"Space '{repo_id}' created.")

    # Upload the trained model to the Hugging Face model space
    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",  # Path to the trained model file
        path_in_repo="best_tourism_package_model_v1.joblib",  # Upload with the same file name
        repo_id=repo_id,  # Repo ID on Hugging Face
        repo_type=repo_type,  # Repo type, in this case 'model'
    )
    print(f"Model uploaded to Hugging Face at: {repo_id}")
