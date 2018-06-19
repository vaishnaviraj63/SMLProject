from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('btc_merged.csv',header=None)
#print(df.head(n=10))
#rows,columns = df.shape
df.columns=['0','1','2','3','4','5','6','7','8','9','10']
#print(df.head())


import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(style='whitegrid', context='notebook')
#cols = ['0', '1', '2', '3', '10']
#sns.pairplot(df[cols], size=2.5)
#plt.show()




from sklearn.cross_validation import train_test_split
X = df.iloc[:, :-1].values
y = df['10'].values

print X

print(y)
# notice we access the column data by passing the column string name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

#Y_predicted = lr.predict(X)

#print(Y_predicted[:10])
#print(y[:10])

#p = df.index[:30]
#q = Y_predicted[:30]
#r = y[:30]


p = df.index[:50]
q = y_test[:50]
r = y_test_pred[:50]


plt.plot(p,q)
plt.plot(p,r)
plt.show()


train_error = mean_squared_error(y_train, y_train_pred)
test_error =  mean_squared_error(y_test, y_test_pred)
print('MSE train: %.3f, test: %.3f' % (train_error,test_error))


print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))




plt.plot(p, q, c= 'blue', marker='o', label='predicted price')
plt.plot(p, r, c= 'lightgreen', marker='s', label= 'true price')
plt.xlabel(' No of days ')
plt.ylabel('target price value')
plt.legend(loc='upper left')
#plt.hlines(y=0, xmin=-100, xmax=500, lw=2, color='red')
#plt.xlim([-100.00,500.00])
plt.show()




from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge_regressor = Ridge(alpha=1.0)
lasso_regressor = Lasso(alpha=1.0)
elastic_regressor = ElasticNet(alpha=1.0, l1_ratio=0.5)
ridge_regressor.fit(X_train, y_train)
y_train_pred_ridge = ridge_regressor.predict(X_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print('MSE train Ridge: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_ridge), mean_squared_error(y_test, y_test_pred_ridge)))
print('R^2 train Ridge: %.3f, test: %.3f' % ( r2_score(y_train, y_train_pred_ridge), r2_score(y_test, y_test_pred_ridge)))
lasso_regressor.fit(X_train, y_train)
y_train_pred_lasso = lasso_regressor.predict(X_train)
y_test_pred_lasso = lasso_regressor.predict(X_test)
print('MSE train Lasso: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_lasso),mean_squared_error(y_test, y_test_pred_lasso)))
print('R^2 train Lasso: %.3f, test: %.3f' % ( r2_score(y_train, y_train_pred_lasso), r2_score(y_test, y_test_pred_lasso)))
elastic_regressor.fit(X_train, y_train)
y_train_pred_elastic = elastic_regressor.predict(X_train)
y_test_pred_elastic = elastic_regressor.predict(X_test)
print('MSE train elastic: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_elastic),mean_squared_error(y_test, y_test_pred_elastic)))
print('R^2 train elastic: %.3f, test: %.3f' % (     r2_score(y_train, y_train_pred_elastic),     r2_score(y_test, y_test_pred_elastic)))
