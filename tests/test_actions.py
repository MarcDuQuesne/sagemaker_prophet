import logging
import sagemaker
import boto3
import pandas as pd
import sys
import pytest
from sagemaker.workflow.pipeline import Pipeline
from conftest import params

logger = logging.getLogger("tests")


def test_fit(fbprophet_training_estimator):
    fbprophet_training_estimator.fit(
        {"training": f"s3://{params['bucket']}/data/example_retail_sales.csv"}
    )


def test_deployment(endpoint):

    df = pd.read_csv(f"s3://{params['bucket']}/data/example_retail_sales.csv")
    payload = df.iloc[0:2]

    response = boto3.client(
        "sagemaker-runtime", region_name=params["region"]
    ).invoke_endpoint(
        EndpointName=params["endpoint_name"],
        Body=payload.to_json(),
        ContentType="json",
    )
    predictions = response["Body"].read().decode("utf-8")
    assert predictions is not None, "Predictions is None."


if __name__ == "__main__":

    import sys

    sys.exit(pytest.main([__file__]))
