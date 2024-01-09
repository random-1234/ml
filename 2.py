import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

data=pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target
print(data.head()) #displaying 5 rows of data
count=data.info()
print(count)
#to print number of null values
print(data.isnull().sum())
data.plot()
plt.show()
#cov matrix and corr matrix
cov_mat=data.cov(numeric_only=True)
corr_mat=data.corr(numeric_only=True)
print(cov_mat)
print(corr_mat)
#train and test model
X=data.drop(["PRICE"],axis=1)
y=data["PRICE"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.09,random_state = 42)
model=SGDRegressor()
model.fit(X_train,y_train)
#predicting values
y_pred=model.predict(X_test)
print(y_pred)
#accuracy and its graph
mse = mean_squared_error(y_test, y_pred)
a = 1 - (mse / np.var(y_test))
# a=accuracy_score(y_test,y_pred)
print(f"the accuracy of the model is : {a} ")