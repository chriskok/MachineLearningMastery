# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'neg_log_loss'
# scoring = 'accuracy'
# scoring = 'neg_mean_squared_error'
# scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("Logloss: %.3f (%.3f)") % (results.mean(), results.std()))
# print(results)

# Confusion Matrix
y_pred = cross_val_predict(model, X, Y, cv=kfold)
conf_mat = confusion_matrix(Y,y_pred)
print('\nConfusion Matrix:\n', conf_mat)

# Classification Report
target_names = ['class 0', 'class 1']
print('\nClassification Report:\n', classification_report(Y, y_pred, target_names=target_names))