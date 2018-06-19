
"""
Pre-prcoessing for data to compute the features
"""
import csv
import sys
f = open("test.out", 'w')
sys.stdout = f
print "Output for Neural Network: \n"
total_data = []
label = []

data = []

with open('btc_final_day.csv') as myFile:
    reader = csv.reader(myFile)
    for row in reader:
        data.append(row)

for i in range(len(data)):
    temp = []
    if i <= len(data) - 11:
        for j in range(10):
            temp.append(float(data[i + j][2]))
        label.append(float(data[i + j + 1][2]))
        # temp.append(0 if float(temp[10]) < temp[9] else 1)
        total_data.append(temp)

print total_data
print label

myFile = open('btc_merged.csv', 'wb')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(total_data)


"""
Code for Nerual Network based regression Using sci-kit learn library
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import sklearn
X_train, X_test, y_train, y_test = train_test_split(total_data, label, test_size=0.20)
# We are suing the MLP Regressor function of the Sci-KitLearn library
# We use the Adam stochastic gradient as the solver, hidden layer size as ten and a learning rate if 0.01
clf = MLPRegressor(hidden_layer_sizes=(10,10), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                   random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                   early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf.fit(X_train, y_train)

y_pred = sklearn.model_selection.cross_val_predict(
    estimator=clf,
    X=X_test,
    y=y_test,
)
print y_pred

print clf.best_loss_
print clf.coefs_
print clf.n_layers_
print clf.n_outputs_
print clf.out_activation_


print('Accuracy testing : {:.3f}'.format(clf.score(X_test, y_test)))

predicted_label = clf.predict(X_test)

"""
Code for plotting the ROC curve
"""

import matplotlib.pyplot as plt
import numpy as np

builds = np.array([i for i in range(10)])

fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(builds, predicted_label[0:10], label='PREDICTED LABELS', color='c', marker='o')
ax1.plot(builds, y_test[0:10], label='TURE LABELS', color='g', marker='o')

plt.xticks(builds)
plt.xlabel('Months')

handles, labels = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
ax1.grid('on')

plt.savefig('smooth_plot_ten_days.png')
f.close()
