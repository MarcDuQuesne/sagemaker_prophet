import logging
import sagemaker
import boto3
import pandas as pd
import sys
import pytest
from sagemaker.workflow.pipeline import Pipeline
from conftest import params

logger = logging.getLogger("tests")


def test_pipeline(sm_session, fbprophet_estimator):

    training_step = sagemaker.workflow.steps.TrainingStep(
        name="CustomFBProphetTrain",
        estimator=fbprophet_estimator,
        inputs={"training": f"s3://{params['bucket']}/data/example_retail_sales.csv"},
    )

    pipeline = Pipeline(
        name="Test-FBProphet",
        parameters=[],
        steps=[
            training_step,
        ],
        sagemaker_session=sm_session,
    )

    upsert_response = pipeline.upsert(role_arn=params["sagemaker_role_arn"])
    logger.info(upsert_response)

    execution = pipeline.start()
    execution.wait()
    pipeline_steps = execution.list_steps()
    logger.info(pipeline_steps)


if __name__ == "__main__":

    import sys

    sys.exit(pytest.main([__file__]))
