# Stat Summary
import pandas as pd
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
description = data.describe()
print(data.head(n=5))
print(data.shape)
print(data.dtypes)
print(description)
print(data.corr(method='pearson'))
