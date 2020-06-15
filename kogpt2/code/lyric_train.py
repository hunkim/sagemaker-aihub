# Author: https://github.com/MrBananaHuman/KorGPT2Tutorial
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from new_tokenizer import MyTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# Based on github.com/pytorch/examples/blob/master/word_language_model
import argparse
import math
import os
from shutil import copy
import time
import torch
import torch.nn as nn

# Run outside of SM for quick testing
if 'SM_MODEL_DIR' not in os.environ:
    print("Running locally?")

    os.environ['SM_MODEL_DIR'] = "./.model_dir"
    os.environ['SM_OUTPUT_DATA_DIR'] = "./.out_data"
    os.environ['SM_CHANNEL_TRAINING']="./.train_data"

    # Make a sample data 
    os.makedirs('./.model_dir', exist_ok=True)
    os.makedirs('./.out_data', exist_ok=True)
    os.makedirs('./.train_data', exist_ok=True)


import subprocess as sp
import os

def get_gpu_memory():
    if not torch.cuda.is_available():
        return 0

    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values

get_gpu_memory()

parser = argparse.ArgumentParser(description='Transformer Language Model')

# Hyperparameters sent by the client are passed as command-line arguments to the script.

parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--save-steps', type=int, default=10_000)
parser.add_argument('--save-total-limit', type=int, default=2)
parser.add_argument('--overwrite-output-dir', type=bool, default=True)
parser.add_argument('--vocab-size', type=int, default=52_000)
parser.add_argument('--max-position-embeddings', type=int, default=514)
parser.add_argument('--num-attention-heads', type=int, default=12)
parser.add_argument('--num-hidden-layers', type=int, default=6)
parser.add_argument('--type-vocab-size', type=int, default=1)
parser.add_argument('--token-max-len', type=int, default=512)

# Data and model checkpoints/otput directories from the container environment
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

args = parser.parse_args()

vocab_file_path = os.path.join(args.data_dir, 'tokenizer/vocab.json')
merge_file_path = os.path.join(args.data_dir,'tokenizer/merges.txt')
model_file = os.path.join(args.data_dir,'KorGPT-2SampleModel/pytorch_model.bin')

tokenizer = MyTokenizer(vocab_file_path, merge_file_path)
tokenizer.save(args.model_dir) # Save it to model dir for generation

config = GPT2Config(vocab_size=52000)
model = GPT2LMHeadModel(config)
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
model.to("cpu").eval() # Memory
get_gpu_memory()

ATTR_TO_SPECIAL_TOKEN = ['<song>', '</song>']

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_song = tokenizer.convert_tokens_to_ids('<song>')
e_song = tokenizer.convert_tokens_to_ids('</song>')

class LyricDataSet(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
        
    def split_songs(self, lines):
        songs = []
        single_song = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(single_song) > 5:
                    songs.append(single_song)
                single_song = []
            else:
                single_song.append(line)
        return songs
    
    def load_data(self):
        lyric_file = open(self.file_path, 'r', encoding='utf-8')
        lyric_lines = lyric_file.readlines()
        lyric_file.close()
        
        song_list = self.split_songs(lyric_lines)
        for song in song_list:
            song_data = ['<song>']
            for line in song:
                tokenized_line = ['<s>'] + tokenizer.tokenize(line) + ['</s>']
                if len(song_data) + len(tokenized_line) < 1024:
                    song_data += tokenized_line
                else:
                    break
            song_data += ['</song>']
            padded_song_data = song_data + ['<pad>'] * (1024 - len(song_data))
            self.data.append(torch.tensor(tokenizer.convert_tokens_to_ids(padded_song_data)).unsqueeze(0))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item


        
lyric_file_path = os.path.join(args.data_dir,'lyric_data/preprocessed_data.txt')
lyric_data = LyricDataSet(lyric_file_path)
lyric_data.load_data()
lyric_data_loader = DataLoader(lyric_data, 
    batch_size=args.batch_size, shuffle=True)


optimizer = AdamW(model.parameters(), lr=1e-4, correct_bias=True)

count = 0
avg_loss = (0.0, 0.0)
model = model.to(device)

for epoch in range(args.epochs):
    for data in lyric_data_loader:
        optimizer.zero_grad()
        data = data.transpose(1,0)
        data = data.to(device)
        
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]
        loss = loss.to(device)
        loss.backward()
        avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
        optimizer.step()
        count+=1        

    print('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(epoch, count, loss, avg_loss[0] / avg_loss[1]))
    get_gpu_memory()

# store thetorch.save(model.state_dict(), 
torch.save(model.state_dict(), 
    os.path.join(args.model_dir, 'lyric_model.bin'))


saved_files = [str(x) for x in Path(args.model_dir).glob("**/*")]
print(saved_files)

# quick testing 
bos = tokenizer.convert_tokens_to_ids('<s>')
eos = tokenizer.convert_tokens_to_ids('</s>')
pad = tokenizer.convert_tokens_to_ids('<pad>')
unk = tokenizer.convert_tokens_to_ids('<unk>')

def add_special_tokens_(model, tokenizer):
    orig_num_tokens = tokenizer.get_vocab_size()
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    num_added_tokens = len(ATTR_TO_SPECIAL_TOKEN)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens + 1)

add_special_tokens_(model, tokenizer)
b_song = tokenizer.convert_tokens_to_ids('<song>')
e_song = tokenizer.convert_tokens_to_ids('</song>')

def encoding(text):
    tokens = ['<song>', '<s>'] + tokenizer.tokenize(text)
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

def decoding(ids):
    return tokenizer.convert_ids_to_tokens(ids[0])

input_ids = encoding('하늘을 날아').to(device)

sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=1024, 
    top_k=50, 
    top_p=0.95, 
    eos_token_id=e_song,
    early_stopping=True,
    bad_words_ids=[[unk]]
)
print(decoding(sample_outputs.tolist()))