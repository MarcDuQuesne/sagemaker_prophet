#!/bin/bash

set -e

ecr='196851712332.dkr.ecr.eu-west-1.amazonaws.com'
prophet_version=$(grep 'prophet' requirements.txt | cut -d'=' -f3)

aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin ${ecr}

# build docker image
# $ AWS_PROFILE=developer_dev_tst-301748432165
docker build --network=host -t ${ecr}/prophet:toolkit-inference-${prophet_version} .

# push docker image
# $ AWS_PROFILE=developer_dev_tst-301748432165
docker push ${ecr}/prophet:toolkit-inference-${prophet_version}
