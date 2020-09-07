import unicodedata
import re
import numpy as np
import tensorflow as tf
import random
import os
import io
import time
from re import finditer
import argparse


"""## Preprocessing helper functions"""

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

# split lowercase followed by uppercase letter into words  
def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return ' '.join([m.group(0) for m in matches])


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())
  
  # creating a space between a word and the punctuation following it
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  
  # replacing everything with space except (a-z, A-Z, "0-9", ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z0-9?.!,¿']+", " ", w)
  w = w.strip()
  
  # adding a start and an end token to the sentence
  w = '<start> ' + w + ' <end>'
  return w


def preprocess_rdf(w):

  w = unicode_to_ascii(w.strip())
  w = camel_case_split(w)

  # creating a space between a word and the punctuation following it
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, 0-9, ".", "?", "!", "," , "/", "\" )
  w = re.sub(r"[^a-zA-Z0-9?.!,¿<>\\\\////']+", " ", w)
  w = w.strip()

  # adding a start and an end token to the sentence
  w = '<start> ' + w.lower() + ' <end>'
  return w


"""## Dataset creation and loading functions"""
def create_dataset(path):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = []
  for l in lines:
    rdf, text = l.split('\t')
    word_pairs.append([preprocess_sentence(text), preprocess_rdf(rdf)])

  return zip(*word_pairs)

def tokenize(lang, char=False):

  if char:
      lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
           filters='',char_level=True)
  else:
      lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
           filters='', oov_token = '<UNK>')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
  return tensor, lang_tokenizer

def load_dataset(path, num_examples=None, char=False):
  # creating cleaned input, output pairs
  targ_lang, inp_lang  = create_dataset(path)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang, char)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden, carry):
    x = self.embedding(x)
    output, state, c = self.lstm(x, initial_state = [hidden, carry])###################
    return output, state, c

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

  def initialize_carry_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))



"""## Define a custom Bahdanau attention layer"""

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):

    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


"""## Define Decoder architecture"""

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, carry, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state, carry = self.lstm(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)

    return x, state, carry, attention_weights

  def initialize_carry_state(self):
    return tf.zeros((self.batch_sz, self.dec_units))


def evaluate(sentence):

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] if i in list(inp_lang.word_index.keys()) else inp_lang.word_index['<UNK>'] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = ''

  # Encoder inputs
  enc_hidden = tf.zeros((1, units))
  enc_carry = tf.zeros((1, units))
  enc_out, enc_hidden, enc_carry = encoder(inputs, enc_hidden, enc_carry)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
  dec_carry = [tf.zeros((1, units))]

  #without teacher forcing
  for t in range(max_length_targ):
    predictions, dec_hidden, dec_carry, attention_weights = decoder(dec_input,
                                                         dec_hidden, dec_carry,
                                                         enc_out)
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence
    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence

def translate(rdf, print_output=False):
  result, rdf = evaluate(rdf)
  if print_output:
    print('\nInput: %s' % (rdf))
    print('\nPredicted translation: {}'.format('<start>' + result))
  return result, rdf
  


def restore_checkpoint():
  try:
    encoder.load_weights('./checkpoints/lstm/full_encoder')
    decoder.load_weights('./checkpoints/lstm/full_decoder')
    print('Model weights loaded.')
  except Exception as e:
    print(e, ' occured while loading weights.')
    pass

def gen_output(out_reference, out_hypothesis):
    with open(out_reference, encoding = 'UTF-8',  mode='a+') as ref, open(out_hypothesis, encoding = 'UTF-8',  mode='a+') as hypo :
        for index, (txt, rdf) in enumerate(zip(txts, trips)):
            pred, input_rdf = translate(rdf, print_output=False)
            ref.write(txt + '\n')
            hypo.write(pred + '\n')
            if index % 10 ==0:
                print('Progress :', round(index/len(trips)*100,2) , '%')


def get_args():

    parser = argparse.ArgumentParser(description="Main Arguments")
    
    parser.add_argument(
      '--test_path', default='./data/test_data.txt', type=str, required=False,
      help='Path to test data')

    parser.add_argument(
      '--checkpoint_path', default='./checkpoints/generator', type=str, required=False,
      help='Path to checkpoint files')

    parser.add_argument(
        '--out_reference', default='./data/transformer_output/reference_file.txt', type=str, required=False,
        help='address of text file where real output will be saved.')

    parser.add_argument(
        '--out_hypothesis', default='./data/transformer_output/hypothesis_file.txt', type=str, required=False,
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


if __name__ == '__main__':
  units=1024
  batch_size=32
  embedding_dim=256
  txts, trips = create_dataset('./data/test_data.txt')
  print('- | Tensorizing dataset...')
  input_tensor, target_tensor, inp_lang, targ_lang = load_dataset('./data/test_data.txt')
  max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
  vocab_inp_size = len( inp_lang.word_index) + 1
  vocab_tar_size = len(targ_lang.word_index) + 1

  encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
  decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
  print('-| Loading model weights...')
  restore_checkpoint()
  print('- | Translating input...')

  gen_output('./data/lstm_output/reference.txt', './data/lstm_output/hypothesis.txt')


