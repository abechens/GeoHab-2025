import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load a sample dataset (e.g., Iris dataset)
from sklearn.datasets import load_iris, load_wine, load_diabetes
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Feature importance
feature_importances = model.feature_importances_
for i, feature in enumerate(data.feature_names):
    print(f"Feature importance of {feature}: {feature_importances[i]}")