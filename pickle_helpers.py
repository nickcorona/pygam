import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file