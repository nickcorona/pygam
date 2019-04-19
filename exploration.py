import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from pygam import datasets

df = datasets.chicago(return_X_y=False)
df.info()

msno.bar(df)
plt.show()  # 3 features with missing values

msno.matrix(df)
plt.show()  # 2 features have missing values that aren't distributed randomly. 1 feature has too many missing values to keep.

msno.heatmap(df)
plt.show() # low nullity correlation +1

# plot distribution of deaths

sns.distplot(df['death']) # roughly normal distribution
plt.show()
