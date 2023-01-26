import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
import time
from utils import create_overlay, labels_only

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global processor
    global device
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    models = ["shi-labs/oneformer_ade20k_swin_tiny", "shi-labs/oneformer_ade20k_dinat_large"]
    timestart = time.time()
    processor = OneFormerProcessor.from_pretrained(models[1])
    model = OneFormerForUniversalSegmentation.from_pretrained(models[1]).to(device)
    print("Model loaded in: ", time.time() - timestart)

# Inference is ran for every server call
# Reference your preloaded global model variable here.

def inference(model_inputs:dict, img_bytes, debug = False) -> dict:
    global model
    # Parse out your arguments
    # Get file from request 
    timestart = time.time()
    task = model_inputs.get("task", "semantic")
    mode = model_inputs.get("mode", "overlay")
    shadow_strength = model_inputs.get("shadow_strength", "1")

    #model = model_inputs.get("model", "shi-labs/oneformer_ade20k_swin_tiny")
    input_img = Image.open(BytesIO(img_bytes))

    #If "seed" is not sent, we won't specify a seed in the call
    if img_bytes == None:
        return {'message': "No image provided"}
        
    # Run the model
    inputs = processor(input_img, [task], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[input_img.size[::-1]])[0]
    predicted_semantic_map_np = predicted_semantic_map.cpu().numpy().astype(np.uint8)
    
    seg = labels_only(predicted_semantic_map_np)

    if (debug):
        seg.save("labelsSeg.png")

    buffered = BytesIO()
    seg.save(buffered,format="PNG", optimize=True, quality=50)

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print("Inference time: ", time.time() - timestart)
    # Return the results as a dictionary
    if mode=="segmentation":
        return {'image_base64': image_base64}

    else:
        overlay = create_overlay(np.array(input_img), predicted_semantic_map_np,floor_id=3, shadow_strength= float(shadow_strength))
        buffered = BytesIO()
        overlay.save(buffered,format="PNG", optimize=True, quality=50)

        if debug:
            overlay.save("overlay2.png", "PNG")

        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {'image_base64': image_base64, 'overlay_base64': overlay_base64}

if __name__ == "__main__":
    init()
    with open("input/room.jpg", "rb") as f:
        img_bytes = f.read()
    inference({"task": "semantic", "mode": "overlay"}, img_bytes, debug=True)