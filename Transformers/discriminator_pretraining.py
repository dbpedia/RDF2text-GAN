# -*- coding: utf-8 -*-
"""
DETAILS:
Script to pretrain the discriminator. The synthetic dataset used here is simplistic,
as we randomly pair input triples with target text as fake instances, and true triple
text pairs as real instances.

The discriminator model has 2 inputs, one for an encoded tensor representing the
input triples, and another for its corresponding text, encoded and tensorized similarly.
Each input layer is followed by its own embedding layer and transformer block,
the outputs of the transformer blocks are then concatenated, before being passed
to the dense classification layer.

The model learns to distinguish real triple text pairs from fake ones.

REQUIREMENTS:
- path to data file (use provided data files, or generate from provided parsers)
- path to vocab file (use provided subword file or generate from pretraining script)

Optional requirements are to do with discriminator model parameters
See help for information
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import time
import numpy as np
import argparse

import pretraining
from pretraining import *
from transformer_discriminator import *


def get_args():

  parser = argparse.ArgumentParser(description="Main Arguments")

  parser.add_argument(
    '--file_path', default='./data/parsed_xml_data.txt', type=str, required=True,
    help='Path to training data')

  parser.add_argument(
    '--vocab_filename', default='./data/subword_tokenizer', type=str, required=True,
    help='Path to subwords vocab file')

  parser.add_argument(
    '--embed_dim', default=32, type=int, required=False,
    help='Embedding dimension')

  parser.add_argument(
    '--num_heads', default=2, type=int, required=False,
    help='Number of attention heads')

  parser.add_argument(
    '--ff_dim', default=32, type=int, required=False,
    help='Number of feed forward layer neurons')

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  
  args = get_args()

  tokenizer_txt =  tfds.features.text.SubwordTextEncoder.load_from_file(args.vocab_filename)
  vocab_size = tokenizer_txt.vocab_size+2

  print('-| Creating generator dataset...')
  train_dataset, _ = create_generator_dataset(args.file_path, tokenizer_txt)

  print('-| Creating simple discriminator dataset...')
  Xr_train, Xr_test, Xt_train, Xt_test, y_train, y_test, max_len = create_discriminator_dataset_2(train_dataset)


  print('-| Loading model...')
  discriminator = TransformerDiscriminator2(vocab_size ,
                                            maxlen=max_len,
                                            embed_dim=args.embed_dim,
                                            num_heads=args.num_heads,
                                            ff_dim=args.ff_dim)
  
  discriminator.compile("adam", "binary_crossentropy", metrics=["accuracy"])

  print('-| Initiating training session...')
  history = discriminator.fit(
      {"rdf": Xr_train,
       "txt": Xt_train},
      {"real_prob": y_train},
      epochs=5,
      batch_size=32,
      shuffle=True,
      validation_data=(
                        {"rdf": Xr_test,
                        "txt": Xt_test},
                       {"real_prob": y_test}
                      )
                  )

  discriminator.save_weights('./checkpoints/discriminator/discriminator_weights.h5')

