from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import datasets
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

print(data.head())
x=data["sepal length (cm)"]
y=data["target"]
plt.xlabel("sepal length (cm)")
plt.ylabel("target")
plt.plot(x,y)
plt.show()
X=data.drop(["target"],axis=1)
y=data["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)
model=KNeighborsClassifier()
model.fit(X,y)
y_pred=model.predict(X_test)
print(y_pred)
a=accuracy_score(y_test,y_pred)
print(f"the accuracy is : {a}")