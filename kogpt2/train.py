import mxnet as mx
from kogpt2.mxnet_kogpt2 import get_mxnet_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer


import argparse
import math
import os
from shutil import copy
import time


if mx.context.num_gpus() > 0:
  ctx = mx.gpu()
else:
  ctx = mx.cpu()

parser = argparse.ArgumentParser(description='PyTorch KoGPT2')

# For testing only
if 'SM_MODEL_DIR' not in os.environ:
    os.environ['SM_MODEL_DIR'] = os.environ['SM_CHANNEL_TRAINING'] = os.environ['SM_OUTPUT_DATA_DIR'] = "./cache"
    

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

model, vocab = get_mxnet_kogpt2_model(ctx=ctx, cachedir=args.model_dir)
print(model)
print(vocab)
print("DONE")