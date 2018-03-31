# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# load dataset
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]


# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('SVM', svm.SVR()))
models.append(('CART', tree.DecisionTreeRegressor()))
models.append(('RFR', RandomForestRegressor(n_estimators=5)))
models.append(('RFR2', RandomForestRegressor(n_estimators=20)))
models.append(('GBR', GradientBoostingRegressor()))


# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Plotting results
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
# data = pandas.Series(data=results, index=)
# print(names)
# print(results)
# scatter_matrix(data)
plt.boxplot(results, labels=names)
# ts = pd.Series(np.random.randn(10), index=pd.date_range('1/1/2000', periods=10))
# ts = ts.cumsum()
# ts.plot()


# ????
# df = pd.DataFrame(data=results, index=names)
# df = df.T
# df = df.cumsum()
# df.plot()
# plt.show()

# df2 = pd.DataFrame(np.random.rand(10, 8), columns=names)
# df2.plot.bar()
plt.show()
