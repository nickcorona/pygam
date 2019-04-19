import numpy as np
import pandas as pd
import missingno as msno
import pickle
from sklearn import preprocessing


df = pd.read_csv('RejectStatsA.csv', header=1)

# remove spaces with underscores
new_column_names = []
for column in df.columns:
    new_column_names.append(column.lower().replace(' ', '_').replace('-', '_'))

df.columns = new_column_names
df = df.dropna()

df['debt_to_income_ratio'] = df['debt_to_income_ratio'].apply(
    lambda x: x.replace('%', '')).astype(float) / 100
df['employment_length'] = df['employment_length'].apply(lambda x: x.replace('years', '').
                                                        replace('year', '').
                                                        replace('<', '').
                                                        replace('+', '')).astype(int)
df['application_date'] = df['application_date'].astype('datetime64')
df['year'] = df['application_date'].dt.year
df['month'] = df['application_date'].dt.month
df['day'] = df['application_date'].dt.day

df = df.drop(['loan_title', 'zip_code', 'application_date', 'policy_code'], axis=1)

X = df[['risk_score', 'debt_to_income_ratio', 'state', 'employment_length', 'year', 'month', 'day']]
y = df['amount_requested']

le = preprocessing.LabelEncoder()
X['state'] = le.fit_transform(df['state'])
X = X.to_numpy()
y = y.to_numpy()

with open('X.pickle', 'wb') as handle:
    pickle.dump(X, handle)

with open('y.pickle', 'wb') as handle:
    pickle.dump(y, handle)