from pygam import datasets
import pandas as pd

df = datasets.chicago(return_X_y=False)
df.to_pickle('data/raw/chicago.pickle')