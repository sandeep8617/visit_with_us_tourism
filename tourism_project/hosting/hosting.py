# Import necessary libraries
from huggingface_hub import HfApi
import os

# Initialize the Hugging Face API client with authentication token
# This token allows access to Hugging Face for uploading files
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload a local folder to a specified Hugging Face repository
api.upload_folder(
    folder_path="tourism_project/deployment",     # Path to the local folder that contains your files to upload
    repo_id="sandeep8617/visit-with-us-tourism-project",          # Target repository on Hugging Face where the files will be uploaded
    repo_type="space",                          # Can be "dataset", "model", or "space" depending on the type of repository
    path_in_repo="",                            # Optional: You can specify a subfolder path inside the repo where the files should be placed
)
