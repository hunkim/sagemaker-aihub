from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# Based on github.com/pytorch/examples/blob/master/word_language_model
import argparse
import math
import os
from shutil import copy
import time
import torch
import torch.nn as nn

# Run outside of SM
if 'SM_MODEL_DIR' not in os.environ:
    print("Running locally?")

    os.environ['SM_MODEL_DIR'] = "./.model_dir"
    os.environ['SM_OUTPUT_DATA_DIR'] = "./.out_data"
    os.environ['SM_CHANNEL_TRAINING']="./.train_data"

    # Make a sample data 
    os.makedirs('./.model_dir', exist_ok=True)
    os.makedirs('./.out_data', exist_ok=True)
    os.makedirs('./.train_data', exist_ok=True)

    f = open(os.path.join(os.environ['SM_CHANNEL_TRAINING'],"sample.txt"), "a")
    f.write("Now the file has more content!\nThis is cool\n")
    f.close()


parser = argparse.ArgumentParser(description='Transformer Language Model')

# Hyperparameters sent by the client are passed as command-line arguments to the script.
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', type=bool, default=False,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

# Data and model checkpoints/otput directories from the container environment
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])


args = parser.parse_args()

print(args)

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(args.data_dir).glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save(args.model_dir)

tokenizer = ByteLevelBPETokenizer(
    os.path.join(args.model_dir, "vocab.json"),
    os.path.join(args.model_dir, "merges.txt"),
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("Mi estas Julien."))
print(tokenizer.encode("Mi estas Julien.").tokens)

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir, max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
model.num_parameters()
# => 84 million parameters

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    #file_path=os.path.join(args.data_dir,"oscar.eo.txt"),
    file_path=paths[0],
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=os.path.join(args.model_dir, "train_out"),
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

print("Training...")
trainer.train()

print("Saving model")
trainer.save_model(args.model_dir)

