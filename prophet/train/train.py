"""
A simple class to implement the API requested by sagemaker
"""

import os
import logging
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json


logger = logging.getLogger("sagemaker_prophet")

def train(args):

    # TODO what formats do we support?
    training_files = list(args.training.glob('*.csv'))
    assert len(training_files) > 0, f"No csv files found in {args.training}."
    assert len(training_files) == 1, "More than one training file not supported ATM."
    data = pd.read_csv(training_files[0])

    # Fitting
    ph_model = Prophet()
    ph_model.fit(data)

    # Serializing the model.
    args.model_dir.mkdir(exist_ok=True)

    with (args.model_dir / 'fb_prophet.json').open('w') as fout:
        fout.write(model_to_json(ph_model))

    logger.info(f'Model saved to {args.model_dir}/fb_prophet.json.')


if __name__ == "__main__":

    parser = ArgumentParser()

    # reads input channels training and testing from the environment variables
    parser.add_argument("--yearly-seasonality", type=int, default=1)
    parser.add_argument("--training", type=Path, default=os.environ.get("SM_CHANNEL_TRAINING", None))
    parser.add_argument("--validation", type=Path, default=os.environ.get("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("SM_MODEL_DIR", None))

    train(parser.parse_args())
