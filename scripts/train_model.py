import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Load the data
data_path = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data/combined_features.csv"
data = pd.read_csv(data_path)

# Verify the dataset is loaded correctly
print(data.head())  # Optional: Print the first few rows of the dataset

# Step 2: Separate features and labels
X = data.drop(columns=["label"])  # Features
y = data["label"]  # Labels

# Step 3: Handle class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("Class distribution after oversampling:", Counter(y_resampled))

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Step 7: Save the trained model
model_path = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/models/random_forest_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# Step 8: Visualize feature importance
importance = model.feature_importances_
plt.barh(range(len(importance)), importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature Index")
plt.show()
