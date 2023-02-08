import torch
import base64
from io import BytesIO
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
import time
from utils import *

def init():
    global model
    global processor
    global device
    global transform

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Modelo segmentation
    models = ["shi-labs/oneformer_ade20k_swin_tiny", "shi-labs/oneformer_ade20k_dinat_large"]
    timestart = time.time()
    processor = OneFormerProcessor.from_pretrained(models[1])
    model = OneFormerForUniversalSegmentation.from_pretrained(models[1]).to(device)
    print("Segmentation model loaded in: ", time.time() - timestart)


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
    vanishing_method = model_inputs.get("vanishing_method", "2")

    input_img = Image.open(BytesIO(img_bytes))

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
        #overlay = create_overlay(np.array(input_img), predicted_semantic_map_np,floor_id=3, shadow_strength= float(shadow_strength))
        overlay = create_visualizer(predicted_semantic_map_np)
        buffered = BytesIO()
        overlay.save(buffered,format="PNG", optimize=True, quality=50)

        if debug:
            overlay.save("overlay2.png", "PNG")
            visual = create_overlay(np.array(input_img), predicted_semantic_map_np,floor_id=3, shadow_strength= float(shadow_strength))
            visual.save("visual.png", "PNG")

        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        angles = get_angle(np.array(input_img), method=vanishing_method, angle='radians')

        if (debug):
            print("Angles: ", angles)

        return {'image_base64': image_base64, 'overlay_base64': overlay_base64, 'pitch': angles[0], 'yaw': angles[1]}

if __name__ == "__main__":
    init()
    with open("input/deluxe.jpg", "rb") as f:
        img_bytes = f.read()
    inference({"task": "semantic", "mode": "overlay"}, img_bytes, debug=True)
