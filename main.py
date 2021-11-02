import pickle

with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)