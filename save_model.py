import shutil
import bentoml

local_model_dir = '/data/Qwen3.6-35B-A3B-FP8'

with bentoml.models.create(
    name='Qwen3.6-35B-A3B', # Name of the model in the Model Store
) as model_ref:
    # Copy the entire model directory to the BentoML Model Store
    shutil.copytree(local_model_dir, model_ref.path, dirs_exist_ok=True)
    print(f"Model saved: {model_ref}")
