FROM python:3.8

ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_ROOT_USER_ACTION=ignore

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt update && apt install -y default-jre
# copy the script inside the container
COPY serve.py /opt/ml/code/serve.py
COPY dockerd_entrypoint.py /opt/ml/code/dockerd_entrypoint.py
ENV PYTHONPATH /opt/ml/code
ENV HANDLER_SERVICE "/opt/ml/code/serve.py:handle"
ENTRYPOINT [ "python", "/opt/ml/code/dockerd_entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]