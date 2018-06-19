import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Reading csv file
data = pd.read_csv('btc_merged.csv')

# Separating the feature and target variables
X = data.ix[:, 0:-1].values
y = data.ix[:, -1].values

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Defining parameters for GridSearchCV
Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
Max_iter = [10, 100, 1000]
param_grid = {'C': Cs, 'max_iter': Max_iter}

# Building the classifier
clf1 = LogisticRegression()
clf = GridSearchCV(clf1, param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=None), scoring='roc_auc', verbose=2, refit=True)
clf.fit(X_train, y_train)

print clf.best_params_

# Testing phase
y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)

# Accuracy
accuracy_score = metrics.accuracy_score(y_test, y_pred)
print "\nAccuracy is: ", accuracy_score

# Average precision
average_precision = metrics.average_precision_score(y_test, y_pred)
print "Average precision is: ", average_precision

# ROC Score
roc_auc_score = metrics.roc_auc_score(y_test, probs[:, 1])
print "ROC Score is: ", roc_auc_score, "\n"

# Plot precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='b')
plt.fill_between(recall, precision, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:, 1], pos_label=1)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % metrics.auc(fpr, tpr))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print "Confusion matrix: \n", confusion_matrix, "\n"

# Classification Report
classification_report = classification_report(y_test, y_pred)
print "Classification Report: \n", classification_report
