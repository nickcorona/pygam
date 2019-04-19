from pygam import LinearGAM, s, f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

X = pd.read_pickle("data/processed/X.pickle")
y = pd.read_pickle("data/processed/y.pickle")

lam = np.logspace(-3, 5, 5)
lams = [lam] * 6
search_space = 1
for array in lams:
    search_space *= len(array)
print(search_space)

# randomized grid search
gam_grid = LinearGAM()
gam_grid.gridsearch(X, y, lam=lams)

with open("models/gam_random_grid_search.pickle", "wb") as handle:
    pickle.dump(gam_grid, handle)

gam_grid.summary()  # (798, 118.1757), (4096, 117.7854)

## plotting
plt.figure()
fig, axs = plt.subplots(1, 6)

titles = ["pm10median", "pm25median", "o3median", "so2median", "time", "tmpd"]
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
