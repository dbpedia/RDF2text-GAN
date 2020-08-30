# -*- coding: utf-8 -*-
"""
DETAILS:
This script is used to pre-train the generator, before adverserial training.

Model is a standard sequence to sequence transformer, usually used in the
context of language translation. (See transformer_generator script for details)

During pretraining uses teacher forcing, i.e. showing the model the correct
previous tokens when predicting the next token in a sequence, regardless of
what the model predicted at previous time steps.

REQUIRES:
- path to training data
- path to subwords vocab file (provided in data directory)

Other arguments include checkpoint directory, and model parameters
See help for info on these arguments

"""

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import re
from re import finditer
from pretraining import *
from transformer_generator import *
import argparse


def get_args():

  parser = argparse.ArgumentParser(description="Main Arguments")

  parser.add_argument(
    '--file_path', default='./data/parsed_xml_data.txt', type=str, required=True,
    help='Path to training data')

  parser.add_argument(
    '--vocab_filename', default='./data/subword_tokenizer', type=str, required=True,
    help='Path to subwords vocab file')

  parser.add_argument(
    '--checkpoint_path', default='./checkpoints/train', type=str, required=False,
    help='Path to checkpoint files')

  parser.add_argument(
    '--num_layers', default=6, type=int, required=False,
    help='Number of transformer layers')

  parser.add_argument(
    '--dmodel', default=512, type=int, required=False,
    help='Number of neurons per layer')

  parser.add_argument(
    '--dff', default=256, type=int, required=False,
    help='Embedding dimension')

  parser.add_argument(
    '--num_heads', default=16, type=int, required=False,
    help='Number of attention heads')

  parser.add_argument(
    '--dropout_rate', default=0.2, type=float, required=False,
    help='Fraction of neurons to drop during training')


  parser.add_argument(
    '--EPOCHS', default=10, type=int, required=False,
    help='Number of epochs model should train for')

  args = parser.parse_args()

  return args


def loss_function(real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)  
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)



if __name__ == '__main__':

  def train_model():
    
    for epoch in range(EPOCHS):
      start = time.time()
      train_loss.reset_states()
      train_accuracy.reset_states()
      
      for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        
        if batch % 50 == 0:
          print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))
          
      if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
      print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))
      print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      
  args = get_args()
  file_path = args.file_path
  checkpoint_path = args.checkpoint_path
  num_layers = args.num_layers
  d_model = args.dmodel
  dff = args.dff
  num_heads = args.num_heads
  dropout_rate = args.dropout_rate
  EPOCHS = args.EPOCHS

  tf.autograph.set_verbosity(2)

  print('\n -| Preparing dataset for training...\n')

  tokenizer_txt =  tfds.features.text.SubwordTextEncoder.load_from_file(args.vocab_filename)
  train_dataset, _ = create_generator_dataset(file_path, tokenizer_txt)


  print('-| Initializing model parameters...\n')
  target_vocab_size = tokenizer_txt.vocab_size + 2
  input_vocab_size = target_vocab_size
  learning_rate = CustomSchedule(d_model)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                       epsilon=1e-9, amsgrad=True)


  print('-| Loading model and weights...\n')
  transformer = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, target_vocab_size, 
                            pe_input=input_vocab_size, 
                            pe_target=target_vocab_size,
                            rate=dropout_rate)
  
  ckpt = tf.train.Checkpoint(transformer=transformer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

  print('-| Initiating training session...\n')
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

  train_model()








