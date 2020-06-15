# Author: https://github.com/MrBananaHuman/KorGPT2Tutorial
import json
import logging
import os

import torch

from transformers import pipeline
from new_tokenizer import MyTokenizer

from transformers import GPT2LMHeadModel, GPT2Config
import torch

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)

ATTR_TO_SPECIAL_TOKEN = ['<song>', '</song>']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

def model_fn(model_dir):
    logger.info('Loading the model.')

    vocab_file_path = os.path.join(model_dir, 'vocab.json')
    merge_file_path = os.path.join(model_dir, 'merges.txt')
    model_file_path = os.path.join(model_dir,'lyric_model.bin')

    tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
    bos = tokenizer.convert_tokens_to_ids('<s>')
    eos = tokenizer.convert_tokens_to_ids('</s>')
    pad = tokenizer.convert_tokens_to_ids('<pad>')
    unk = tokenizer.convert_tokens_to_ids('<unk>')

    config = GPT2Config(vocab_size=52003, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)

    model = GPT2LMHeadModel(config)

    model.load_state_dict(torch.load(model_file_path, map_location=device), strict=False)
    model.to(device)

    return model, tokenizer


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


def predict_fn(input_data, model_tokenizer):
    logger.info('Generating text based on input parameters.')
    model, tokenizer = model_tokenizer

    def add_special_tokens_(model, tokenizer):
        orig_num_tokens = tokenizer.get_vocab_size()
        tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

    add_special_tokens_(model, tokenizer)
    b_song = tokenizer.convert_tokens_to_ids('<song>')
    e_song = tokenizer.convert_tokens_to_ids('</song>')
    unk = tokenizer.convert_tokens_to_ids('<unk>')


    def encoding(text):
        tokens = ['<song>', '<s>'] + tokenizer.tokenize(text)
        return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

    def decoding(ids):
        return tokenizer.convert_ids_to_tokens(ids[0])

    input_ids = encoding(input_data['text']).to(device)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=20, 
        top_k=10, 
        top_p=0.95, 
        eos_token_id=e_song,
        early_stopping=True,
        bad_words_ids=[[unk]]
    )
    
    return decoding(sample_outputs.tolist())

# For testing, make sure you run the infer.py and store the model outputs to S3
if __name__ == '__main__':
    model_dir="./.model"

    os.makedirs(model_dir, exist_ok=True)

    # Please run train and zip and store the model directory to S3
    import boto3
    s3 = boto3.client('s3')
    s3.download_file('sagemaker-us-west-2-294038372338', 
    'pytorch-training-2020-06-15-00-00-33-479/output/model.tar.gz', 'model.tar.gz')

    import tarfile
    tar = tarfile.open("model.tar.gz")
    tar.extractall(model_dir)
    tar.close()

    print("Loading model done!")

    input = {'text': "오늘은 "}
    model = model_fn(model_dir)
    print(predict_fn(input, model))

    import shutil
    shutil.rmtree(model_dir)
