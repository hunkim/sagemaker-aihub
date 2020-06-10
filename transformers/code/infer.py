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
    return {'model': fill_mask, 'corpus': "TBA"}


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
    corpus = model['corpus']
    fill_mask = model['model']

    return fill_mask(input_data['text'])

if __name__ == '__main__':
    model_dir="./.model"

    # Downlaod model
    # s3://sagemaker-us-west-2-294038372338/pytorch-training-2020-06-10-11-15-27-506/output/model.tar.gz
    import boto3
    s3 = boto3.client('s3')
    s3.download_file('sagemaker-us-west-2-294038372338', 
    'pytorch-training-2020-06-10-11-15-27-506/output/model.tar.gz', 'model.tar.gz')

    import tarfile
    tar = tarfile.open("model.tar.gz")
    tar.extractall(model_dir)
    tar.close()

    input = {'text': "La suno <mask>."}
    model = model_fn(model_dir)
    print(predict_fn(input, model))
