import pandas as pd
import pickle

df = pd.read_pickle('data/raw/chicago.pickle')
print('Read chicago dataset.')

# fill missing values with median. might not be good
# enough for `pm10median` and `pm25median`
features_with_missing = ['pm10median', 'pm25median', 'so2median']

for feature in features_with_missing:
    df[feature] = df[feature].fillna(df[feature].median())

print('Filled missing values.')

y = df['death'].to_numpy()
X = df.iloc[:, 1:].to_numpy()

print('Created feature matrix, `X`, and target vector, `y`.')

with open('data/processed/X.pickle', 'wb') as handle:
    pickle.dump(X, handle)

print("Wrote `X` to 'data/processed/X.pickle'")

with open('data/processed/y.pickle', 'wb') as handle:
    pickle.dump(y, handle)

print("Wrote `y` to 'data/processed/y.pickle'")
