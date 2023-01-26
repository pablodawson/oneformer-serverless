import banana_dev as banana
import base64
import json
from io import BytesIO
from PIL import Image
import time

image_path = "input/room3.jpg"

with open(image_path, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

api_key = "b6b9bb42-7432-4a66-8a3f-90f9f4b69c38"
model_key = "de69cd53-7ddf-45c2-b202-bdcd00124ada"

dictionary = {"image": im_b64, "task": "semantic"}
timestart = time.time()

out = banana.run(api_key, model_key, dictionary)

print("Time taken: ", time.time() - timestart)

image_byte_string = out["modelOutputs"][0]["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("outputweb.jpg")
