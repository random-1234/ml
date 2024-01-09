import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

print("First 5 rows of the dataset:")
print(data.head())

print("\nChecking for null values:")
print(data.isnull().sum())

data.plot()
plt.show()

print("\nCovariance Matrix:")
print(data.cov(numeric_only=True))

print("\nCorrelation Matrix:")
print(data.corr(numeric_only=True))

X = data.drop('PRICE', axis=1)
y = data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)