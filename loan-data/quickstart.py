import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pygam.datasets import wage
from pygam import LinearGAM, s, f


X, y = wage()
gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
gam.summary()
print(gam.lam)

lam = np.logspace(-3, 5, 5)
lams = [lam] * 3
gam.gridsearch(X, y, lam=lams)
gam.summary()

lams = np.random.rand(100, 3)
lams = lams * 6 - 3
lams = np.exp(lams)
random_gam =  LinearGAM(s(0) + s(1) + f(2)).gridsearch(X, y, lam=lams)
random_gam.summary()


gam.statistics_['GCV'] < random_gam.statistics_['GCV']

list(gam.statistics_.keys())

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()