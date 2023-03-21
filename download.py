# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import os
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    models = ["shi-labs/oneformer_ade20k_swin_tiny", "shi-labs/oneformer_ade20k_dinat_large"]
    processor = OneFormerProcessor.from_pretrained(models[1])
    model = OneFormerForUniversalSegmentation.from_pretrained(models[1])
    

if __name__ == "__main__":
    download_model()
