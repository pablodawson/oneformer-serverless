# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
from io import BytesIO
from PIL import Image
import json

image_path = "input/bedroom.png"

def save_img(byte_string, i):
    image_encoded = byte_string.encode('utf-8')
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    image.save(i+"output.png")

with open(image_path, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8") #base64

payload = json.dumps({"image": im_b64, "task": "semantic", "vanishing_method": 1})

res = requests.post('http://localhost:8000/', data = payload)

save_img(res.json()["image_base64"], "seg_")
save_img(res.json()["overlay_base64"], "overlay_")

print(res.json()["pitch"])
print(res.json()["yaw"])

#https://stackoverflow.com/questions/29104107/upload-image-using-post-form-data-in-python-requests