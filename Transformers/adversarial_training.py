# -*- coding: utf-8 -*-
"""
## Adverserial training script

DETAILS:
Trains the discriminator and the generator in an adversarial fashion.
This example is inspired by tensorflow's advanced turorials
to make a DCGAN, found here: https://www.tensorflow.org/tutorials/generative/dcganµ


REQUIREMENTS:
- .txt data file (provided under data directory)
- subword_tokenizer file (provided under data directory)

All other arguments pertain to model parameters
See help for more info


PROBLEM:
The script runs into the famous 'No gradients provided for any variable problem'
Both losses, for discriminator and generator, are computed. The discriminator model
is able to backprop gradients and update weights, only the generator fails to do so.

Array of solutions tried (including coding the training step differently, manually
watching accessed variables, etc.

## Solution pending...

"""

import os
import argparse
import time
import io
import unicodedata
import re
from re import finditer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

from pretraining import *
from transformer_generator import *
from transformer_discriminator import *



"""## Loss and metrics"""

def discriminator_loss(real_output, fake_output):

    '''
    Quantifies discriminator's ability to distinguish real sequences from fakes.
    It compares the discriminator's predictions on real sequences to an array of 1s,
    and the discriminator's predictions on fake (generated) sequences
    to an array of 0s.
    '''
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



def generator_loss(fake_output):

    '''
    Quantifies generator's ability to trick the discriminator. 
    If the generator is doing well, discriminator will classify 
    fake sequences as real (or 1). We thus compare the discriminators
    decisions on the generated sequences to an array of 1s.
    '''
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    fake_output = tf.convert_to_tensor(fake_output, dtype=tf.float32)
    loss_ = loss_object(tf.ones_like(fake_output,dtype=tf.float32), fake_output)
    return  loss_



"""## Define adversarial training step"""

def train_step(inp, tar):
    # targets shifted by 1 index position
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    #Get encoding, combined and decoding masks
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # Initialize Generator gradient tape
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Get prediction probabilities from generator
        predictions, _ = generator(inp, tar_inp, 
                             True, 
                             enc_padding_mask, 
                             combined_mask, 
                             dec_padding_mask)
        # Get predicted sequences for batch
        batch_pred = tf.argmax(predictions, axis=-1)

        # Pad predicted batch
        batch_pred = tf.keras.preprocessing.sequence.pad_sequences(batch_pred, padding='post',
                                                                   value=0, maxlen=tar.shape[-1])
        # Get discriminator's predictions of real & generated output
        disc_preds_real = discriminator([inp, tar], training=True)
        disc_preds_fake = discriminator([inp, batch_pred], training=True)

        # Calculate loss using discriminator and generator loss functions
        d_loss = discriminator_loss(disc_preds_real, disc_preds_fake)
        g_loss = generator_loss(disc_preds_fake)

    # Get discriminator gradients and apply using optimizer
    disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_weights)
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
    
    # Get generator gradients and apply using optimizer
    gen_grads = gen_tape.gradient(g_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))



"""## Define training function"""

def train():
  '''
  Function to initialize training process
  Prints Generator and discriminator loss during training
  '''
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


def get_args():

  parser = argparse.ArgumentParser(description="Main Arguments")

  parser.add_argument(
    '--file_path', default='./data/parsed_xml_data.txt', type=str, required=True,
    help='Path to training data')

  parser.add_argument(
    '--vocab_filename', default='./data/subword_tokenizer', type=str, required=True,
    help='Path to subwords vocab file')

  parser.add_argument(
    '--gen_checkpoint_path', default='./checkpoints/generator', type=str, required=False,
    help='Path to checkpoint files')

  parser.add_argument(
    '--disc_checkpoint_path', default='./checkpoints/discriminator/discriminator_weights.h5', type=str, required=False,
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

  parser.add_argument(
    '--batch_size', default=16, type=int, required=False,
    help='Batch size of dataset')

  args = parser.parse_args()

  return args




#Generator params
if __name__ == '__main__':

    args = get_args()

    file_path = args.file_path

    generator_checkpoint = args.gen_checkpoint_path
    discriminator_checkpoint = args.disc_checkpoint_path

    batch_size = args.batch_size

    EPOCHS = args.EPOCHS
    num_layers = args.num_layers
    d_model = args.dmodel
    dff = args.dff
    num_heads = args.num_heads
    dropout_rate = args.dropout_rate


    tokenizer_txt =  tfds.features.text.SubwordTextEncoder.load_from_file(args.vocab_filename)
    input_vocab_size = target_vocab_size = tokenizer_txt.vocab_size + 2
    
    train_dataset, tokenizer_txt = create_generator_dataset(file_path, tokenizer_txt, BATCH_SIZE=batch_size)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    learning_rate = CustomSchedule(d_model)


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')


    print('-| Loading Generator model and weights...\n')
    generator = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, target_vocab_size, 
                            pe_input=input_vocab_size, 
                            pe_target=target_vocab_size,
                            rate=dropout_rate)

    ckpt = tf.train.Checkpoint(transformer=generator)
    ckpt_manager = tf.train.CheckpointManager(ckpt, generator_checkpoint, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('-| Latest checkpoint restored!!')

    print('-| Loading Discriminator model and weights...\n')

    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator = TransformerDiscriminator2(tokenizer_txt.vocab_size+2, maxlen=250)
    if os.path.exists(discriminator_checkpoint):
        discriminator.load_weights(discriminator_checkpoint)

    train()



