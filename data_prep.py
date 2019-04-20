import pandas as pd
import pickle

df = pd.read_pickle('data/raw/chicago.pickle')
print('Read chicago dataset.')

# dropping `pm25median' because of too many missings
# dropping `o3median` because it is insignificant to the GAM model
drop_features = ['pm25median', 'o3median']
df = df.drop(drop_features, axis=1).dropna(how='any')

print(f'Dropped {drop_features} and then any rows with missing values.')

y = df['death'].to_numpy()
X = df.iloc[:, 1:].to_numpy()

print('Created feature matrix, `X`, and target vector, `y`.')

with open('data/processed/X.pickle', 'wb') as handle:
    pickle.dump(X, handle)

print("Wrote `X` to 'data/processed/X.pickle'.")

with open('data/processed/y.pickle', 'wb') as handle:
    pickle.dump(y, handle)

print("Wrote `y` to 'data/processed/y.pickle'.")
