import pandas as pd
import pickle

df = pd.read_pickle('data/raw/chicago.pickle')
print('Read chicago dataset.')

df = df.drop('pm25median', axis=1).dropna(how='any')

print('Dropped `pm25median` and then any rows with missing values.')

y = df['death'].to_numpy()
X = df.iloc[:, 1:].to_numpy()

print('Created feature matrix, `X`, and target vector, `y`.')

with open('data/processed/X.pickle', 'wb') as handle:
    pickle.dump(X, handle)

print("Wrote `X` to 'data/processed/X.pickle'.")

with open('data/processed/y.pickle', 'wb') as handle:
    pickle.dump(y, handle)

print("Wrote `y` to 'data/processed/y.pickle'.")
