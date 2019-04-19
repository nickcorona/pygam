import pandas as pd
from pygam import LinearGAM, s, f

X = pd.read_pickle('X.pickle')
y = pd.read_pickle('y.pickle')

gam = LinearGAM(s(0) + s(1) + f(2) + s(3) + s(4) + s(5) + s(6)).fit(X, y)
gam.summary()