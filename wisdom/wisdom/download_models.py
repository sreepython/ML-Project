"""
pip install torch transformers
sudo apt-get install tmux

tmux attach-session -t model_download_session

"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def download_model(model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(save_dir, 'download_log.txt')
    log_fp = open(log_file, 'a')

    def log(message):
        print(message)
        log_fp.write(message + '\n')

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    log(f"Downloading model '{model_name}' to '{save_dir}'.")

    # Start model download
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    log("Model download completed.")

if __name__ == "__main__":
    # List of models to download
    models_to_download = [
        {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "description": "8x7B Instruct Model"},
        # Add more models as needed
    ]

    for model_info in models_to_download:
        model_name = model_info["name"]
        model_description = model_info["description"]

        # Use the current directory and model name
        save_directory = os.path.join(os.getcwd(), model_name.replace("/", "-"))

        # Run the download in the background using tmux
        tmux_session_name = f"{model_name}_download_session"
        tmux_command = f"python download_models.py {model_name}"

        # Start a new tmux session in the background
        os.system(f"tmux new-session -d -s {tmux_session_name} '{tmux_command}'")

        print(f"Downloading {model_description} in the background. You can monitor the progress using 'tmux attach-session -t {tmux_session_name}'.")
