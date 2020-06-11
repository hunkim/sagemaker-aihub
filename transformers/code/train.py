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

    #f = open(os.path.join(os.environ['SM_CHANNEL_TRAINING'],"sample.txt"), "a")
    #f.write("Now the file has more content!\nThis is cool\n")
    #f.close()


parser = argparse.ArgumentParser(description='Transformer Language Model')

# Hyperparameters sent by the client are passed as command-line arguments to the script.

parser.add_argument('--num-train-epochs', type=int, default=1)
parser.add_argument('--per-gpu-train-batch-size', type=int, default=64)
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

paths = [str(x) for x in Path(args.data_dir).glob("**/*.txt")]
print("data files")
print(paths)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(
    files=paths, 
    vocab_size=52_000, 
    min_frequency=2, 
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>"]
)

# Need to save it to model dir for inference
tokenizer.save(args.model_dir)

tokenizer = ByteLevelBPETokenizer(
    os.path.join(args.model_dir, "vocab.json"),
    os.path.join(args.model_dir, "merges.txt")
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>"))
)
tokenizer.enable_truncation(max_length=args.token_max_len)

print(tokenizer.encode("Nay, but speak not."))
print(tokenizer.encode("Nay, but speak not.").tokens)

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=args.vocab_size,
    max_position_embeddings=args.max_position_embeddings,
    num_attention_heads=args.num_attention_heads,
    num_hidden_layers=args.num_hidden_layers,
    type_vocab_size=args.type_vocab_size
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(
    args.model_dir, 
    max_len=args.token_max_len
)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
model.num_parameters()
# => 84 million parameters

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    #file_path=os.path.join(args.data_dir,"oscar.eo.txt"),
    file_path=paths[0], # Get the first file
    block_size=128
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=args.output_data_dir,
    overwrite_output_dir=args.overwrite_output_dir,
    num_train_epochs=args.num_train_epochs,
    per_gpu_train_batch_size=args.per_gpu_train_batch_size,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True
)

print("Training...")
trainer.train()

print("Saving model...")
trainer.save_model(args.model_dir)

saved_files = [str(x) for x in Path(args.model_dir).glob("**/*")]
print(saved_files)


