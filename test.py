# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
from io import BytesIO
from PIL import Image
import json

image_path = "input/room.jpg"

with open(image_path, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

payload = json.dumps({"image": im_b64, "task": "semantic"})

res = requests.post('http://localhost:8000/', data = payload)

image_byte_string = res.json()["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")
#https://stackoverflow.com/questions/29104107/upload-image-using-post-form-data-in-python-requests