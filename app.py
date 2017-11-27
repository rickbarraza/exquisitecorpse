import json
import eventlet
import time
import base64

from flask import Flask, render_template, session, request, current_app
from flask_socketio import SocketIO, emit, disconnect

eventlet.monkey_patch()

import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shhhh!'

socketio = SocketIO(app, async_mode="eventlet")

import pickle
import os
filename = 'data/queries_lexicon.pkl'


def get_lexicon_lookup(lexicon):
    lexicon_lookup = { idx: lexicon_item for lexicon_item, idx in lexicon.items()}
    lexicon_lookup[0] = "" #map 0 padding to empty string
    lexicon_lookup[793] = ""
    print("LEXICON LOOKUP SAMPLE:")
    print(list(lexicon_lookup.items())[500:510])
    return lexicon_lookup

def tokens_to_ids(all_tokens, lexicon):
    ids = [[ lexicon[token] if token in lexicon else lexicon['<UNK>'] for token in token_line] for token_line in all_tokens]
    return ids

'''
from keras import backend as K
from importlib import reload
import os

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        #reload(K)
        #assert K.backend() == backend

set_keras_backend("theano")
'''

from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

def create_model(seq_input_len, n_input_nodes, n_embedding_nodes, n_hidden_nodes, stateful=False, batch_size=None):

    input_layer = Input(batch_shape=(batch_size, seq_input_len), name='input_layer')
    
    embedding_layer = Embedding(input_dim=n_input_nodes,
                               output_dim=n_embedding_nodes,
                               mask_zero=True, name='embedding_layer')(input_layer)
    
    gru_layer1 = GRU(n_hidden_nodes,
                    return_sequences=True,
                    stateful=stateful,
                    name='hidden_layer1')(embedding_layer)
    
    gru_layer2 = GRU(n_hidden_nodes,
                    return_sequences=True,
                    stateful=stateful,
                    name='hidden_layer2')(gru_layer1)
    
    output_layer = TimeDistributed(Dense(n_input_nodes, activation="softmax"),
                                  name='output_layer')(gru_layer2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    
    return model
    
lexicon = None
lexicon_lookup = None
predictor_model = None

import numpy 

def generate_ending(idx_seq):
    end_of_sent_tokens = [".", "!","/",";","?",":"]
    generated_ending = []
    
    #with current_app.graph():
    if len(idx_seq) == 0:
        return [3]
    
    for word in idx_seq:
        p_next_word = current_app.predictor_model.predict(numpy.array(word)[None, None])[0,0]
        
    while not generated_ending or current_app.lexicon_lookup[next_word] not in end_of_sent_tokens:
        next_word = numpy.random.choice(a=p_next_word.shape[-1], p=p_next_word)

        if next_word != 1:
            generated_ending.append(next_word)
            p_next_word = current_app.predictor_model.predict(numpy.array(next_word)[None, None])[0,0]
        
    current_app.predictor_model.reset_states()
    return generated_ending



import spacy

def text_to_tokens(lines):
    tokens = [ [word.lower_ for word in current_app.encoder(line)] for line in lines]
    return tokens

def linePredict(test_line):
    input_line = tokens_to_ids( text_to_tokens(test_line), current_app.lexicon)
    generated_ending = generate_ending(input_line[0])
    if not generated_ending[0] == 2:
        generated_ending = " ".join([current_app.lexicon_lookup[word] if word in current_app.lexicon_lookup else "" for word in generated_ending])
    
    return generated_ending


import re
import random


def getStarters():
    with open('data/frank_starters.txt') as f:
        text = f.read()
        frank_sentences = re.split(r' *[\.\!][\'"\)\]]* *', text)
        
        for i in range(len(frank_sentences)):
            frank_sentences[i] = frank_sentences[i].replace("\n", " ") + "."

    print("total starter sentences", len(frank_sentences))
    return frank_sentences    


def getStarter():
    wordCount = 0

    totalStarters = len(current_app.frank_sentences)

    while wordCount < 12:
        starter = random.randint(0, totalStarters)
        rawfirst = current_app.frank_sentences[starter]
        splitted = rawfirst.split()
        wordCount = len(splitted)

    stopWord = splitted[12]
    stopIndex = rawfirst.find(stopWord)
                            
    if stopIndex > 3:
        return rawfirst[:stopIndex]
    else:
        return rawfirst


@app.before_first_request
def initialize_AI():

    print("---------------------------------------\n")
    print("INITIALIZING A.I.")
    current_app.encoder = spacy.load('en')
    current_app.lexicon = pickle.load( open(filename, "rb"))
    current_app.lexicon_lookup = get_lexicon_lookup(current_app.lexicon)
    current_app.predictor_model = create_model(seq_input_len=1,
                        n_input_nodes=len(current_app.lexicon) + 1,
                        n_embedding_nodes=300,
                        n_hidden_nodes = 500,
                        stateful=True,
                        batch_size=1)
    current_app.predictor_model.load_weights('corpse_weights5.h5')

    current_app.frank_sentences = getStarters()
    print("First starter: ", current_app.frank_sentences[0])
    print("---------------------------------------\n")

        

@app.route('/')
def index():
    return render_template('exquisite_corpse.html')


@socketio.on('connect', namespace='/eq')
def app_connect():
    socketio.emit('model_ready', { 'status':"connected",}, namespace='/eq')


@socketio.on('line_submit', namespace='/eq')
def line_submit(data):

    print("\n---------------------------------------")
    print("LINE SUBMITTED: ", data)

    input_line = ""

    if len(data) < 3:
        print("getting starter...")
        input_line = getStarter()
        generated_ending = input_line + " " + linePredict(input_line)
    else:
        print("using data sent...")
        input_line = data
        generated_ending = linePredict(input_line)

    print("generated_ending: ", generated_ending)
    print("---------------------------------------\n")

    socketio.emit('line_append', { 'new_line':generated_ending}, namespace='/eq')

@socketio.on('disconnect', namespace='/eq')
def app_diconnect():
    print("CLIENT DISCONNECTED")

if __name__ == '__main__':
    socketio.run( app, port=5002, debug=True)
