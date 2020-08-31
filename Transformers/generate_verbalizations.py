'''
DETAILS:

This script loads the final transformer model and weights,
and writes predictions to text files. The script generates
two text files, one for the reference and another for the
predicted hypothesis. Each line of the text files correspond
to the real and predicted outputs, respectively.

EVALUATION:
The text files can be directly used to generate automatic evaluation scores,
using the following repository: - https://github.com/WebNLG/GenerationEval

REQUIRES:
- path a data file (file obtained from the xml or json parser scripts)
- path to subword tokenizer (Provided in data directory, can be created using pretraining script)
- path to model checkpoint dir (provided under model_weights directory)
- path to output text file, wherever desired

OPTIONAL:
- Optional arguments only need to be used in case the initial model parameters
   are altered before training, otherwise defaults apply
  (Consult 'help' for information on optional parameters)

'''

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from pretraining import *
from transformer_generator import *
import argparse


def get_args():

    parser = argparse.ArgumentParser(description="Main Arguments")
    
    parser.add_argument(
      '--test_path', default='./data/test_data.txt', type=str, required=True,
      help='Path to test data')

    parser.add_argument(
      '--vocab_filename', default='./data/subword_tokenizer', type=str, required=True,
      help='Path to subword tokenizer file')

    parser.add_argument(
      '--checkpoint_path', default='./checkpoints/generator', type=str, required=False,
      help='Path to checkpoint files')

    parser.add_argument(
        '--out_reference', default='./data/reference_file.txt', type=str, required=False,
        help='address of text file where real output will be saved.')

    parser.add_argument(
        '--out_hypothesis', default='./data/hypothesis_file.txt', type=str, required=False,
        help='address of text file where generated output will be saved.')

    parser.add_argument(
      '--num_layers', default=6, type=int, required=False,
      help='Path to test data')

    parser.add_argument(
      '--d_model', default=512, type=int, required=False,
      help='Path to test data')

    parser.add_argument(
      '--dff', default=256, type=int, required=False,
      help='Path to test data')

    parser.add_argument(
      '--num_heads', default=16, type=int, required=False,
      help='Number of attention heads')

    parser.add_argument(
      '--dropout_rate', default=0.2, type=float, required=False,
      help='Fraction of neurons to drop during training')
    
    args = parser.parse_args()

    return args


def load_model(checkpoint_path):
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
    return transformer
      

def evaluate(inp_sentence, transformer, tokenizer_txt):
  # Magic number, arbitrarily chosen
  # Model usually performs worse on larger sequences
  MAX_LENGTH=250
  encoder_input = tf.expand_dims(inp_sentence, 0)
  decoder_input = [tokenizer_txt.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_txt.vocab_size+1:
      return tf.squeeze(output, axis=0)
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def render_preds(t, predicted_sentence, print_write=False):
    real = decode_text(t, tokenizer_txt)
    preds = decode_text(predicted_sentence, tokenizer_txt)
    if print_write:
        print('Real : ', real)
        print('Predicted : ', preds )
    return real, preds


def gen_output(test_data, transformer, tokenizer_txt, out_reference, out_hypothesis):
    with open(out_reference, encoding = 'UTF-8',  mode='a+') as ref, open(out_hypothesis, encoding = 'UTF-8',  mode='a+') as hypo :
        for index, (r_batch,t_batch) in enumerate(test_dataset):
            for (r, t) in zip(r_batch, t_batch):
                predicted_sentence = evaluate(r, transformer, tokenizer_txt)
                real, preds = render_preds(t, predicted_sentence)
                ref.write(real + '\n')
                hypo.write(preds + '\n')
                if index % 2 ==0:
                    print('Progress :', round(index/len(list(test_dataset))*100,2) , '%')
                    



if __name__=='__main__':
    args = get_args()
    test_path = args.test_path
    vocab_filename = args.vocab_filename
    checkpoint_path = args.checkpoint_path
    out_reference = args.out_reference
    out_hypothesis = args.out_hypothesis
    num_layers = args.num_layers
    d_model = args.d_model
    dff = args.dff
    num_heads = args.num_heads
    dropout_rate = args.dropout_rate
    
    tokenizer_txt =  tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
    test_dataset, _ = create_generator_dataset(test_path, tokenizer_txt)
    target_vocab_size = tokenizer_txt.vocab_size + 2
    input_vocab_size = target_vocab_size
    transformer = load_model(checkpoint_path)
    gen_output(test_dataset, transformer, tokenizer_txt, out_reference, out_hypothesis)



    
