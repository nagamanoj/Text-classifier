# Text-classifier

A simple RNN(LSTM) network to train a text classifier to detect three classes in data set. 
My current data set is specific to one problem but this network can be reused for any text

Initial step in pre-processing text data for a network to learn is done using pre-trained 
word2vec model from Google Brain Team. 
This same network can be used as an sentiment analyser if the last dense layer is replaced 
with output shape having one value. 


