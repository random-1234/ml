# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Import dataset
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()
data = pd.DataFrame(data= np.c_[california_housing.data, california_housing.target], columns= california_housing.feature_names + ['target'])

# Step 2: Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Check for null values
print("\nChecking for null values:")
print(data.isnull().sum())

# Step 4: Visualize data
# For example, let's visualize the relationship between two variables (MedInc and target)
plt.scatter(data['MedInc'], data['target'])
plt.xlabel('MedInc')
plt.ylabel('Target')
plt.title('Relationship between MedInc and Target')
plt.show()

# Step 5: Obtain covariance and correlation values
covariance_matrix = data.cov()
correlation_matrix = data.corr()

print("\nCovariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Step 6: Train and test model
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Apply Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Print the accuracy
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R-squared Value:", r2)