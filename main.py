import numpy
import keras
from utils.preprocess import bag_of_words, load_data
words, labels, docs = load_data("intents.json")

model = keras.models.load_model("model/model.h5")

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