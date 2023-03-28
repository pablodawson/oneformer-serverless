# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import os
import torch
import time
from huggingface_hub import hf_hub_url

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    models = ["shi-labs/oneformer_ade20k_swin_tiny", "shi-labs/oneformer_ade20k_dinat_large"]
    processor = OneFormerProcessor.from_pretrained(models[1])

    url1 = "https://huggingface.co/spaces/pablodawson/convert_oneformer/resolve/main/model.safetensors"
    url2 = "https://huggingface.co/spaces/pablodawson/convert_oneformer/resolve/main/config.json"
    
    # Download model weights
    torch.hub.download_url_to_file(url1, "models/model.safetensors")
    torch.hub.download_url_to_file(url2, "models/config.json")

    model = OneFormerForUniversalSegmentation.from_pretrained("models")

if __name__ == "__main__":
    download_model()
