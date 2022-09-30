"""
A simple class to implement the API requested by sagemaker
"""

import os
import logging
from pathlib import Path
import pandas as pd

from prophet.serialize import model_from_json
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
from sagemaker_inference.transformer import Transformer
from sagemaker_inference.default_handler_service import DefaultHandlerService


logger = logging.getLogger("sagemaker_prophet")
ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"


class ModelHandler:
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        model_dir = context.system_properties.get("model_dir")
        logger.info(f'Reading model from {model_dir}.')
        with open(f"{model_dir}/fb_prophet.json", 'r') as fin:
             self.model = model_from_json(fin.read())  # Load model

    def inference(self, data):
        """
        Internal inference methods
        :param data: transformed model input data list
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        return self.model.predict(data)

    def handle(self, input_batch, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        output_batch = []
        for model_input in input_batch:
            model_input = pd.read_json(model_input['body'].decode('utf-8'))
            model_output = self.inference(model_input)
            output_batch.append(model_output.to_json())

        return output_batch


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)