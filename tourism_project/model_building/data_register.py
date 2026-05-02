
# Import required modules from Hugging Face Hub
# HfApi - used to interact with Hugging Face repositories
# create_repo - creates a new dataset repository if it does not exist
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


# Define Hugging Face repository details
# repo_id - unique identifier for dataset repository (username/dataset-name)
# repo_type - specifies the type of repo (dataset in this case)
repo_id = "sandeep8617/visit-with-us-tourism-project"
repo_type = "dataset"

# Initialize Hugging Face API client using authentication token
# Token is securely fetched from environment variables for security reasons
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository already exists in Hugging Face Hub
try:
    # Try to fetch repository information
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using existing repository.")

# If repository does not exist, handle the exception and create a new one
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    
    # Create a new dataset repository in Hugging Face Hub
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repository '{repo_id}' successfully created.")

# Step 2: Upload processed dataset folder to Hugging Face Hub
# This ensures versioned storage of dataset for MLOps pipeline
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
