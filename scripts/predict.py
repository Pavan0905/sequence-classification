import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load new features (sequence features)
new_data_path = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data/sequence_features.csv"
new_data = pd.read_csv(new_data_path)

# Check the number of rows in the DataFrame
print(f"Shape of new_data: {new_data.shape}")

# Check the first few rows
print(new_data.head())

# Example: if you have actual labels, replace random labels
# labels = pd.read_csv("path_to_labels.csv")["label_column"]

# For testing purposes, generating random labels (replace with real labels)
labels = np.random.choice([0, 1], size=new_data.shape[0])

# Add the labels to your features DataFrame
new_data['label'] = labels

# Separate features (X) and labels (y)
X = new_data.drop(columns=["label"])
y = new_data["label"]

# If there is only one sample, proceed without train-test split
if new_data.shape[0] == 1:
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)  # Train the model on the single sample

    # Make predictions (this will not be meaningful with one sample)
    predictions = model.predict(X)
    print("Predictions:", predictions)

else:
    # Split into training and testing sets if there are more than one sample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model to a file
    model_path = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/models/random_forest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    print("Predictions:", predictions)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
