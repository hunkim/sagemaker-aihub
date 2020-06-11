import json
import logging
import os

import torch

from transformers import pipeline


JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    logger.info('Loading the model.')

    fill_mask = pipeline(
        "fill-mask",
        model=model_dir,
        tokenizer=model_dir,
    )
    return fill_mask


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        if 'text' not in input_data:
            raise Exception('\'text\' has to be set.')
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    logger.info('Generating text based on input parameters.')
    return model(input_data['text'])

# For testing, make sure you run the infer.py and store the model outputs to S3
if __name__ == '__main__':
    model_dir="./.model"

    # Please run train and zip and store the model directory to S3
    import boto3
    s3 = boto3.client('s3')
    s3.download_file('sagemaker-us-west-2-294038372338', 
    'pytorch-training-2020-06-11-00-45-34-666/output/model.tar.gz', 'model.tar.gz')

    import tarfile
    tar = tarfile.open("model.tar.gz")
    tar.extractall(model_dir)
    tar.close()

    input = {'text': "Shall we <mask>."}
    model = model_fn(model_dir)
    print(predict_fn(input, model))
