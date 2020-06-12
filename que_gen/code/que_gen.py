import json
import logging
import os

import torch

from transformers import pipeline
import argparse
import glob
import os
import json
import time
import logging
import random
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

class QueGenerator():
  def __init__(self, model_dir="./"):
    self.que_model = T5ForConditionalGeneration.from_pretrained(model_dir+'/t5_que_gen_model/t5_base_que_gen/')
    self.ans_model = T5ForConditionalGeneration.from_pretrained(model_dir+'/t5_ans_gen_model/t5_base_ans_gen/')

    self.que_tokenizer = T5Tokenizer.from_pretrained(model_dir+'/t5_que_gen_model/t5_base_tok_que_gen/')
    self.ans_tokenizer = T5Tokenizer.from_pretrained(model_dir+'/t5_ans_gen_model/t5_base_tok_ans_gen/')
    
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    self.que_model = self.que_model.to(self.device)
    self.ans_model = self.ans_model.to(self.device)
  
  def generate(self, text):
    answers = self._get_answers(text)
    questions = self._get_questions(text, answers)
    output = [{'answer': ans, 'question': que} for ans, que in zip(answers, questions)]
    return output
  
  def _get_answers(self, text):
    # split into sentences
    sents = sent_tokenize(text)

    examples = []
    for i in range(len(sents)):
      input_ = ""
      for j, sent in enumerate(sents):
        if i == j:
            sent = "[HL] %s [HL]" % sent
        input_ = "%s %s" % (input_, sent)
        input_ = input_.strip()
      input_ = input_ + " </s>"
      examples.append(input_)
    
    batch = self.ans_tokenizer.batch_encode_plus(examples, max_length=512, pad_to_max_length=True, return_tensors="pt")
    with torch.no_grad():
      outs = self.ans_model.generate(input_ids=batch['input_ids'].to(self.device), 
                                attention_mask=batch['attention_mask'].to(self.device), 
                                max_length=32,
                                # do_sample=False,
                                # num_beams = 4,
                                )
    dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    answers = [item.split('[SEP]') for item in dec]
    answers = chain(*answers)
    answers = [ans.strip() for ans in answers if ans != ' ']
    return answers
  
  def _get_questions(self, text, answers):
    examples = []
    for ans in answers:
      input_text = "%s [SEP] %s </s>" % (ans, text)
      examples.append(input_text)
    
    batch = self.que_tokenizer.batch_encode_plus(examples, max_length=512, pad_to_max_length=True, return_tensors="pt")
    with torch.no_grad():
      outs = self.que_model.generate(input_ids=batch['input_ids'].to(self.device), 
                                attention_mask=batch['attention_mask'].to(self.device), 
                                max_length=32,
                                num_beams = 4)
    dec = [self.que_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    return dec

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    logger.info('Loading the model.')

    que_generator = QueGenerator(model_dir)

    return que_generator



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
    return model.generate(input_data['text'])

# For testing, make sure you run the infer.py and store the model outputs to S3
if __name__ == '__main__':
    model_dir="./.model"

    # Please run train and zip and store the model directory to S3
    import boto3
    s3 = boto3.client('s3')
    s3.download_file('sagemaker-us-west-2-294038372338', 
    'sagemaker/hunkim-que-gen/model.tar.gz', 'model.tar.gz')

    import tarfile
    tar = tarfile.open("model.tar.gz")
    tar.extractall(model_dir)
    tar.close()

    input = {'text': "Good to see you. My name is Sung Kim."}
    model = model_fn(model_dir)
    print(predict_fn(input, model))
