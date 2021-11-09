FROM pytorch/pytorch:latest
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt
RUN pip3 install opencv-python-headless==4.5.3.56
RUN pip install waitress==2.0.0