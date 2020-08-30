
""""
DETAILS:
This script contains pre-processing and pretraining
     functions for both Generator and Discriminator
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import re
from re import finditer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split


'''
####| Functions to preprocess and clean raw data files
      Function names:
         - unicode_to_asci()
         - camel_case_split()
         - preprocess_sentence()
         - preprocess_rdf()
         
      -> Used to clean data files created by provided parsers
      -> Used in function create_dataset()
'''

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

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?0-9.!,¿']+", " ", w)
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
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿<>0-9////_']+", " ", w)# \\\\
  w = w.strip()
  # adding a start and an end token to the sentence
  w = '<start> ' + w.lower() + ' <end>'
  return w


'''
####| Functions to prepare data for generator model

      Functions:
        - create_dataset()
        - get_vocab_size()
        - decode_text()
        - create_generator_dataset()
        
      -> Used in generator_pretraining.py and discriminator_pretraining.py
'''

def create_dataset(path, split=False):

  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = []

  for l in lines:
    rdf, text = l.split('\t')
    word_pairs.append((preprocess_rdf(rdf), preprocess_sentence(text) ))
    
  if split:
    return zip(*word_pairs)

  return word_pairs

def get_vocab_size(data_list):
  cache=[]
  for sent in data_list:
    for word in sent.split():
      cache.append(word)
  return len(set(cache))

def decode_text(array, tokenizer):
  return tokenizer.decode([i for i in array if 0<i< tokenizer.vocab_size])



def create_generator_dataset(file_path,  tokenizer_txt, BUFFER_SIZE = 20000, BATCH_SIZE = 64, MAX_LEN=None):
  
  def encode(lang1, lang2):
    lang1 = [tokenizer_txt.vocab_size] + tokenizer_txt.encode(
        lang1.numpy()) + [tokenizer_txt.vocab_size+1]

    lang2 = [tokenizer_txt.vocab_size] + tokenizer_txt.encode(
        lang2.numpy()) + [tokenizer_txt.vocab_size+1]
    
    return lang1, lang2


  def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en
  
  def filter_max_length(x, y, max_length=MAX_LEN):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


  rdfs, txts = create_dataset(file_path, split=True)
  dataset = create_dataset(file_path)

  txts = [i[1] for i in dataset]
  rd = [i[0] for i in dataset]

  max_source_len = max([len(tokenizer_txt.encode(i)) for i in rdfs])
  max_targ_len = max([len(tokenizer_txt.encode(i)) for i in txts])
  print('Max source length : ', max_source_len, '\nMax target length : ', max_targ_len)

  dataset = tf.data.Dataset.from_tensor_slices((np.array(rdfs), np.array(txts)))


  train_dataset = dataset.map(tf_encode)
  if MAX_LEN:
      train_dataset = train_dataset.filter(filter_max_length)
  # cache the dataset to memory to get a speedup while reading from it.
  train_dataset = train_dataset.cache()
  train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return train_dataset, tokenizer_txt


'''
####| Functions to prepare data for concatenated RDF-Text discriminator model
      Function names:
        - prep_disc_data()
        - label_disc_data()
        - gen_disc_data()
        - pad_disc_data()
        - create_discriminator_dataset

      -> Each function has a copy with '2' included in the name, denoting that they
         are used to prepare the dataset for the latest discriminator model, with multiple inputs.
'''

def prep_disc_data(rdf_batch, txt_batch):
    rb , tb = rdf_batch.numpy(), txt_batch.numpy()
    copy_tb = np.copy(tb)
    np.random.shuffle(copy_tb)
    fake = np.concatenate((rb, copy_tb), axis=-1)
    real = np.concatenate((rb, tb), axis=-1)
    return fake, real


def label_disc_data(preds, real): 
  fake = np.hstack([preds, np.zeros((len(preds),1), dtype=np.int64)])
  real = np.hstack([real[:len(preds)], np.ones((len(preds),1), dtype=np.int64)])
  all_data = np.vstack([real, fake])
  np.random.shuffle(all_data)
  X, y = all_data[:, :-1], all_data[:,-1]
  return X, y

def gen_disc_data(train_dataset):
  X_, y_ = [],[]
  for (batch, (inp, targ)) in enumerate(train_dataset):
    fake, real = prep_disc_data(inp, targ)
    X, y = label_disc_data(fake, real)
    X_.append(X)
    y_.append(y)
  return X_, y_


def pad_disc_data(X_, y_, max_len):
    X_data, y_data = [], []

    for (x_batch, y_batch) in zip(X_, y_):
      for (X, y) in zip(x_batch, y_batch):
        output = np.pad(X, (0, max_len - len(X)), 'constant')
        X_data.append(output)
        y_data.append(y)


    X_data, y_data = np.array(X_data, dtype=np.int32), np.array(y_data, dtype=np.int32)
    return X_data, y_data


def create_discriminator_dataset(train_dataset):
  X_, y_ = gen_disc_data(train_dataset)

  max_len = max([x.shape[1] for x in X_])
  print('Max sequence size: ',max_len)

  X_data, y_data = pad_disc_data(X_, y_, max_len)
  print('dataset shape: ', X_data.shape, y_data.shape)


  X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
  
  return X_train, X_test, y_train, y_test


'''
####| Functions to prepare data for multi-input discriminator model
'''

def prep_disc_data_2(rdf_batch, txt_batch):
    rb , tb = rdf_batch.numpy(), txt_batch.numpy()
    copy_tb = np.copy(tb)
    np.random.shuffle(copy_tb)
    
    rdf = rb
    fake_txt = copy_tb
    real_txt = tb

    
    fake_labels =  np.zeros((fake_txt.shape[0],1), dtype=np.int64)
    real_labels = np.ones((real_txt.shape[0],1), dtype=np.int64)
    
    all_rdf = np.vstack([rb, rb])
    all_txt = np.vstack([fake_txt, real_txt])
    all_labels = np.vstack([fake_labels, real_labels])
    return all_rdf, all_txt, all_labels

def gen_disc_data_2(train_dataset):
  X_r, X_t, y = [], [], []
  for (batch, (inp, targ)) in enumerate(train_dataset):
    all_rdf, all_txt, all_labels = prep_disc_data_2(inp, targ)
    X_r.append(all_rdf)
    X_t.append(all_txt)
    y.append(all_labels)
  return X_r, X_t, y 



def pad_disc_data_2(X_r, X_t, y, max_len):
    Xr_data, Xt_data, y_data = [], [], []

    for (xr_batch, xt_batch,  y_batch) in zip(X_r, X_t, y):
      for (Xr, Xt, y) in zip(xr_batch, xt_batch,  y_batch):
        
        Xr_ = np.pad(Xr, (0, max_len - len(Xr)), 'constant')
        Xt_ = np.pad(Xt, (0, max_len - len(Xt)), 'constant')
        Xr_data.append(Xr_)
        Xt_data.append(Xt_)
        y_data.append(y)


    Xr_data, Xt_data = np.array(Xr_data, dtype=np.int32), np.array(Xt_data, dtype=np.int32)
    y_data = np.array(y_data, dtype=np.int32)
    
    return Xr_data, Xt_data, y_data


def create_discriminator_dataset_2(train_dataset):
  X_r, X_t, y = gen_disc_data_2(train_dataset)

  max_len = max([x.shape[1] for x in X_r])
  print('Max RDF sequence size: ',max_len)

  max_len_ = max([x.shape[1] for x in X_t])
  print('Max text sequence size: ',max_len)

  max_len = max(max_len_, max_len)

  X_r, X_t, y = pad_disc_data_2(X_r, X_t, y, max_len)
  print('dataset shape: ', X_r.shape, X_t.shape, y.shape)


  Xr_train, Xr_test, Xt_train, Xt_test, y_train, y_test = train_test_split(X_r, X_t, y, test_size=0.1, random_state=42)
  
  return Xr_train, Xr_test, Xt_train, Xt_test, y_train, y_test, max_len





