import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn import datasets
iris = datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
print(iris_df.head())
print(iris_df.count())
print(iris_df.isnull().any()) #to check null values are present or not
print(iris_df.isnull().sum()) #to print number of null values
iris=iris_df
iris.plot() #graph representation
plt.show()

print("\nCovariance Matrix:")
print(iris.cov())
print("\nCorrelation Matrix:")
print(iris.corr())

X=iris.drop(["target"],axis=1) #to train and test model
y=iris["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_pred)
a=accuracy_score(y_test,y_pred)
print(f"the accuracy is : {a}")