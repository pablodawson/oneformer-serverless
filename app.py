import torch
import base64
from io import BytesIO
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
import time
from utils import *
import os

os.environ["SAFETENSORS_FAST_GPU"] = "1"

def init():
    global model
    global processor
    global device

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Modelo segmentation
    models = ["shi-labs/oneformer_ade20k_swin_tiny", "shi-labs/oneformer_ade20k_dinat_large"]
    timestart = time.time()
    processor = OneFormerProcessor.from_pretrained(models[0])
    model = OneFormerForUniversalSegmentation.from_pretrained("models/").to(device)
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
    vanishing_method = model_inputs.get("vanishing_method", "1")

    input_img = Image.open(BytesIO(img_bytes))

    if img_bytes == None:
        return {'message': "No image provided"}
    
    if (input_img.format=="PNG"):
        input_img = input_img.convert("RGB")

    # Run the model
    inputs = processor(resized_img, [task], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    print("Model time: ", time.time() - timestart)

    timestart = time.time()
    if task == "semantic":
        predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[resized_img.size[::-1]])[0]
        predicted_semantic_map_np = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        seg = labels_only(predicted_semantic_map_np)
        #segmentations = get_wall_instances(np.array(input_img), np.array(seg), debug=True)

        if (debug):
            seg.save("labelsSeg.png")
        
    elif task == "panoptic":
        output = processor.post_process_panoptic_segmentation(outputs, target_sizes=[input_img.size[::-1]])
        predicted_semantic_map, info = output[0]["segmentation"], output[0]["segments_info"]

        predicted_semantic_map_np = predicted_semantic_map.cpu().numpy().astype(np.uint8)

        if debug:    
            cv2.imwrite("labelsPan.png", predicted_semantic_map_np)

    

    buffered = BytesIO()
    seg.save(buffered,format="PNG", optimize=True, quality=50)

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print("Output processing time: ", time.time() - timestart)

    

    # Return the results as a dictionary
    if mode=="segmentation":
        return {'image_base64': image_base64}

    else:
        overlay = create_overlay(np.array(input_img), predicted_semantic_map_np,floor_id=3, shadow_strength= float(shadow_strength))
        #overlay = create_visualizer(predicted_semantic_map_np)
        buffered = BytesIO()
        overlay.save(buffered,format="PNG", optimize=True, quality=50)

        if debug:
            overlay.save("overlay2.png", "PNG")
            visual = create_overlay(np.array(input_img), predicted_semantic_map_np,floor_id=3, shadow_strength= float(shadow_strength))
            visual.save("visual.png", "PNG")

        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        timestart = time.time()
        angles = get_angle(np.array(input_img), method=vanishing_method, angle='radians')
        print("Angles time: ", time.time() - timestart)
        if (debug):
           print("Angles: ", angles)

        return {'image_base64': image_base64, 'overlay_base64': overlay_base64, 'pitch': angles[0], 'yaw': angles[1]}

if __name__=="__main__":
    init()
    image = open("input/room.jpg", "rb").read()
    inference({"task": "semantic", "mode": "overlay"}, image, debug=True)