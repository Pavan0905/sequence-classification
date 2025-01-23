import pandas as pd
from sklearn.model_selection import train_test_split

# Load the combined features
data = pd.read_csv("/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data/combined_features.csv")

# Separate features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import pickle

# Save the model to a file
with open("/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to /Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/models/random_forest_model.pkl")


import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
importance = model.feature_importances_

# Sort features by importance
indices = np.argsort(importance)[::-1]
top_features = X.columns[indices][:10]  # Top 10 features

# Plot
plt.figure(figsize=(10, 6))
plt.bar(top_features, importance[indices][:10])
plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Top 10 Feature Importance")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
print(data["label"].value_counts())

# Filter minority class
minority_class = data[data["label"] == 1]

# Duplicate the minority class to create more samples
for _ in range(5):  # Duplicate 5 times
    data = pd.concat([data, minority_class])

# Re-check the label distribution
print(data["label"].value_counts())


