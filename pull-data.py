import os

from pygam import datasets
import pandas as pd

if not os.path.exists("data"):
    os.mkdir('data')
else:
    print('Data folder already exists.')
if not os.path.exists('data/processed'):
    os.mkdir('data/processed')
else:
    print('Data/processed already exists.')
if not os.path.exists('data/raw'):
    os.mkdir('data/raw')
else:
    print('Data/raw already exists.')

if not os.path.exists('data/raw/chicago.pickle'):
    df = datasets.chicago(return_X_y=False)
    df.to_pickle('data/raw/chicago.pickle')
else:
    print('Chicago dataset already exists.')
