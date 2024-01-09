# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Step 1: Import dataset
# Loading Iris dataset from sklearn
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Step 2: Split the data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create an SVM classifier
# Using a Support Vector Classifier (SVC) with a radial basis function (RBF) kernel
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=42)

# Step 4: Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Step 5: Evaluate its performance
y_pred = svm_classifier.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)
