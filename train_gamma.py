import sys
from pygam import GammaGAM, s, f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

X = pd.read_pickle("data/processed/X.pickle")
y = pd.read_pickle("data/processed/y.pickle")
print('Read data.')

lams = np.random.rand(1000, 3) * 8 - 3
lams = np.exp(lams)

# randomized grid search
print('Initialized Gamma GAM.')
gam_grid = GammaGAM(s(0) + s(2) + s(3))
print("Grid searching Gamma GAM's lambdas.")
gam_grid.gridsearch(X, y, lam=lams)

with open(f"models/{sys.argv[1]}.pickle", "wb") as handle:
    pickle.dump(gam_grid, handle)
print('Serialized GAM as pickle.')

print(gam_grid.summary())

# plotting
plt.figure(figsize=(16, 16 / 1.618))
fig, axs = plt.subplots(1, 3)

titles = ["pm10median", "time", "tmpd"]
for i, ax in enumerate(axs):
    XX = gam_grid.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam_grid.partial_dependence(term=i, X=XX))
    ax.plot(
        XX[:, i],
        gam_grid.partial_dependence(term=i, X=XX, width=0.95)[1],
        c="r",
        ls="--",
    )
    if i == 0:
        ax.set_ylim(-30, 30)
    ax.set_title(titles[i])

plt.savefig(f'images/{sys.argv[1]}-partial-dependency-plots.png')
