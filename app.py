import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global processor

    models = ["shi-labs/oneformer_ade20k_swin_tiny", "shi-labs/oneformer_ade20k_dinat_large"]

    processor = OneFormerProcessor.from_pretrained(models[1])
    model = OneFormerForUniversalSegmentation.from_pretrained(models[1])

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict, img_bytes) -> dict:
    global model

    # Parse out your arguments
    # Get file from request 

    task = model_inputs.get("task", "semantic")
    #model = model_inputs.get("model", "shi-labs/oneformer_ade20k_swin_tiny")
    input_img = Image.open(BytesIO(img_bytes))

    #If "seed" is not sent, we won't specify a seed in the call
    if img_bytes == None:
        return {'message': "No image provided"}
    
    if task == None:
        return {'message': "No task provided"}
    
    # Run the model
    with autocast("cuda"):
        inputs = processor(input_img, ["semantic"], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[input_img.size[::-1]])[0]
        predicted_semantic_map_np = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(predicted_semantic_map_np)
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
