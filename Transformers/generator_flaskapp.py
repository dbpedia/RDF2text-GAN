
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from transformer_generator import *
from pretraining import *


app = Flask(__name__)
api = Api(app)


num_layers = 6
d_model = 512
dff = 256
num_heads = 16
dropout_rate = 0.2
tf.autograph.set_verbosity(2)

vocab_filename = './data/subword_tokenizer.subwords'
tokenizer_txt =  tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)

print('-| Initializing model parameters...\n')
target_vocab_size = tokenizer_txt.vocab_size + 2
input_vocab_size = target_vocab_size


print('-| Loading model and weights...\n')
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


transformer.load_weights('weights.h5')



# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class GenerateText(Resource):
    
    def get(self):

        def evaluate(inp_sentence):
            
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


        vocab_filename = './data/subword_tokenizer'
        tokenizer_txt =  tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)

        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        sentence = tokenizer_txt.encode(user_query)
        result = evaluate(sentence)
        
        # create JSON object
        output = {'prediction': result}
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(GenerateText, '/')


if __name__ == '__main__':
    app.run(debug=True)
