# fit a linear regressor using cross validation

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from pickle_helpers import load_pickle
import matplotlib.pyplot as plt

X = load_pickle('data/processed/X.pickle')
y = load_pickle('data/processed/y.pickle')

regr = linear_model.LinearRegression()
regr.fit(X, y)
