# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ENV	 DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

#OpenCV
RUN apt-get install python3-opencv -y

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install lu_vp_detect --no-dependencies
RUN pip3 install natten -f https://shi-labs.com/natten/wheels/cu113/torch1.11/index.html

# We add the banana boilerplate here
ADD server.py .
ADD utils.py .
ADD vanishing_point_detection.py .

EXPOSE 8000

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py
