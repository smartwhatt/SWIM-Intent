import pickle
from keras.layers import Dense
from keras.models import Sequential
import keras
from datetime import datetime
import json
import random
import tensorflow as tf
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
from numpy.__config__ import show

from utils.preprocess import load_data, create_training_data, bag_of_words

# stemmer = LancasterStemmer()

# Load dataset
words, labels, docs = load_data("intents.json")
docs_x, docs_y = docs

# create trainning data
training, output = create_training_data(words, labels, docs_x, docs_y)

# training = []
# output = []
# out_empty = [0 for _ in range(len(labels))]
# for x, doc in enumerate(docs_x):
#     bag = []
#     wrds = [stemmer.stem(w.lower()) for w in doc]
#     for w in words:
#         if w in wrds:
#             bag.append(1)
#         else:
#             bag.append(0)
#     output_row = out_empty[:]
#     output_row[labels.index(docs_y[x])] = 1
#     training.append(bag)
#     output.append(output_row)
# training = numpy.array(training)
# output = numpy.array(output)
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


# model creation
model = Sequential([
    Dense(8, input_shape=(len(training[0]), ), activation="relu"),
    Dense(8, activation="relu"),
    Dense(len(output[0]), activation="softmax")
])

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = tf.optimizers.Adam(lr=0.0083)
metrics = ["accuracy"]

# print("\nCompiling Model...\n")
model.compile(loss=loss, optimizer=optim, metrics=metrics)

model.fit(training, output, epochs=1000,  verbose=2, validation_split=0.2)
model.save(f"model/model.h5")


while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break
    sentence = numpy.array([bag_of_words(inp, words)])
    results = model.predict(sentence)
    results_index = numpy.argmax(results)
    maxindex = numpy.max(results)
    tag = labels[results_index]
    print(tag)
    print(maxindex)
