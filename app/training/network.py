import gensim
import numpy as np
import re
import json
import os
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv

load_dotenv()

print("loading vectors ....... from location " + os.getenv("GOOGLE_VECTORS_LOCATION"))
w2v = gensim.models.KeyedVectors.load_word2vec_format(os.getenv("GOOGLE_VECTORS_LOCATION"), binary=True)


def get_vector(word):
    wordvec = []
    try:
        wordvec = w2v[word]
    except:
        wordvec = np.zeros(300)
        # wordvec = []

    return wordvec


def clean_data(text):
    lower = text.lower()
    try:
        lower = text.lower()
    except:
        print("error in text")
        print(text)
    # lower = 'hello @how are$% HQ'
    lower = re.sub(r"[^a-z ]+", "", lower)
    lower.strip()
    string_array = lower.split(' ')
    final_array = []
    for word in string_array:
        if word:
            vec = get_vector(word)
            if len(vec) > 0:
                final_array.append(get_vector(word))

    return final_array


def padd_data(arr):
    len_arr = []
    # padd to a prefered lenght it might be longest length or mean or standard deviation value.
    prefered_len = 2000
    for each_arr in arr:
        if len(each_arr) < prefered_len:
            print('padding array with zero')
            while len(each_arr) < prefered_len:
                each_arr.insert(0, np.zeros(300))
        elif len(each_arr) > prefered_len:
            print('truncating the array')
            while len(each_arr) > prefered_len:
                each_arr.pop(0)
        len_arr.append(each_arr)
    return len_arr


def prepare_tensor(data):

    embeded_input = []
    embeded_output = []
    for obj in data:
        if isinstance(obj['txt'], str):
            embeded_input.append(clean_data(obj['txt']))
            if obj['val'] == 'Subscription Cancelation':
                embeded_output.append([1, 0, 0])
            elif obj['val'] == 'Customer Question Answered':
                embeded_output.append([0, 1, 0])
            elif obj['val'] == 'Cust Question Escalated to Tier 2':
                embeded_output.append([0, 0, 1])

    padded_embeded_input = padd_data(embeded_input)
    # converting embedings to numpy arrays
    data_tensor = {
        "input": np.array(padded_embeded_input),
        "output": np.array(embeded_output)
    }
    # np.save('tetxt1_in_data', data_tensor["input"], allow_pickle=True, fix_imports=True)
    # np.save('tetxt1_out_data', data_tensor["output"], allow_pickle=True, fix_imports=True)
    print(np.shape(data_tensor["input"]))
    print(np.shape(data_tensor["output"]))
    return data_tensor


def train_model(data_tensor):
    print("got data building network")
    model = keras.Sequential()
    model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=np.shape(data_tensor["input"])[1:]))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(64, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['acc'])
    history = model.fit(data_tensor['input'], data_tensor['output'], epochs=10, validation_split=0.1)
    print(history.history)


def read_data():
    print("reading from file")
    training_file = open('/Users/manoj/Desktop/callData/new1_audio_validation_set.txt', 'r')
    training_set = json.loads(training_file.read())
    training_file.close()
    # train_model(prepare_tensor(training_set))
    prepare_tensor(training_set)


def load_saved():
    print("reading saved numpy arrays")
    data_tensor = {
        "input": np.load('txt_in_data.npy'),
        "output": np.load('txt_out_data.npy')
    }


# read_data()
