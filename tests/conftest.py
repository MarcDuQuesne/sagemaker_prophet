from sagemaker.workflow.steps import ProcessingStep, TrainingStep
import sagemaker
import boto3
from sagemaker.model import Model
from datetime import datetime
import logging
import pytest

logger = logging.getLogger("tests")

params = {
    "region": "eu-west-1",
    "bucket": "test-prophet-custom-image",
    "sagemaker_role_arn": "arn:aws:iam::196851712332:role/service-role/sagemaker-studio-role",
    "training_docker_image_uri": "196851712332.dkr.ecr.eu-west-1.amazonaws.com/prophet:toolkit-training-1.1",
    "inference_docker_image_uri": "196851712332.dkr.ecr.eu-west-1.amazonaws.com/prophet:toolkit-inference-1.1",
    "endpoint_name": "test-FBProphet"
    + datetime.utcnow().strftime(
        "%Y%m%d%H%m%S"
    ),  # MG is it possible to re-use or update an existing cfg?
}


@pytest.fixture
def sagemaker_session(local=True):

    if local:
        return sagemaker.session.LocalSession()
    else:
        boto_session = boto3.Session(region_name=params["region"])
        return sagemaker.session.Session(
            boto_session=boto_session,
            sagemaker_client=boto_session.client("sagemaker"),
            sagemaker_runtime_client=boto_session.client("sagemaker-runtime"),
        )


@pytest.fixture
def fbprophet_training_estimator(sagemaker_session):

    return sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_uri=params["training_docker_image_uri"],
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=f"s3://{params['bucket']}/FBProphet_trained_model/",
        role=params["sagemaker_role_arn"],
        hyperparameters={"yearly-seasonality": 20},
        use_spot_instances=True,
        max_run=3600,
        max_wait=7200,
        checkpoint_s3_uri=f's3://{params["bucket"]}/checkpoints/current_run',
    )


@pytest.fixture
def endpoint(sagemaker_session):

    session = boto3.client("sagemaker", params["region"])

    # Maybe upload the files to s3?
    model_url = f"s3://{params['bucket']}/FBProphet_trained_model/model.tar.gz"

    model = Model(
        image_uri=params["inference_docker_image_uri"],
        model_data=model_url,
        role=params["sagemaker_role_arn"],
        sagemaker_session=sagemaker_session,
    )

    logger.info(f'Using endpoint: [{params["endpoint_name"]}]')

    yield model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=params["endpoint_name"],
    )

    # Delete endpoint
    session.delete_endpoint(EndpointName=params["endpoint_name"])
    logger.info('Deleted Endpoint.')
