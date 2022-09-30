FROM python:3.8

ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_ROOT_USER_ACTION=ignore

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy the script inside the container
COPY train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM /opt/ml/code/train.py