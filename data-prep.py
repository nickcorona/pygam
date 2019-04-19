import pandas as pd
import pickle

df = pd.read_pickle('data/raw/chicago.pickle')

df = df.dropna(how='any')

y = df['death'].to_numpy()
X = df.iloc[:, 1:].to_numpy()

with open('data/processed/X.pickle', 'wb') as handle:
    pickle.dump(X, handle)

with open('data/processed/y.pickle', 'wb') as handle:
    pickle.dump(y, handle)


