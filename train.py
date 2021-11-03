import pickle
import numpy

from utils.preprocess import load_data, create_training_data, bag_of_words
from utils.model import create_model, train_model
# stemmer = LancasterStemmer()

# Load dataset
words, labels, docs = load_data("intents.json")
docs_x, docs_y = docs

# create trainning data
training, output = create_training_data(words, labels, docs_x, docs_y)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


# model creation
model = create_model(len(training[0]), len(output[0]))

train_model(model, training, output, epoch=50, save=True, filename="model/model.h5")


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
