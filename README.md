
 **Sequence Classification Project - Using Random Forest Classifier**

## **Project Overview**
This project implements a **Random Forest Classifier** for classifying biological sequences (e.g., DNA sequences) based on extracted features. The features include **GC content**, **k-mer frequencies**, and **sequence length**, all of which are crucial for understanding biological data and can be applied to various bioinformatics tasks.

### **Key Features:**
- **Feature Extraction**: Extracts relevant features like GC content, k-mer frequencies (k=3), and sequence length from FASTA formatted biological sequences.
- **Random Forest Classifier**: A machine learning model used to classify sequences into categories based on the extracted features.
- **Handling Imbalanced Data**: The dataset is balanced using **Random Oversampling** (SMOTE) to improve model performance on underrepresented classes.

## **Installation**

To run this project on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Pavan0905/sequence-classification.git
    cd sequence-classification
    ```

2. **Install required dependencies**:
    Make sure you have **Python 3** and **pip** installed. Then install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up a virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
    ```

## **Running the Project**

### **Step 1: Feature Extraction**
- The `feature_extraction.py` script extracts features from the input sequences, including:
    - GC content
    - k-mer frequencies (k=3)
    - Sequence length

To run this script, use:
```bash
python scripts/feature_extraction.py
```

#### **Outputs**:
- **Features** extracted are saved as CSV files in the `/data` directory (e.g., `sequence_features.csv`).

### **Step 2: Train Model**
- The `train_model.py` script trains a **Random Forest Classifier** on the extracted features.
    - It balances the data using **Random Oversampling (SMOTE)**.
    - The model is saved as `random_forest_model.pkl` in the `/models` directory.

To run the model training:
```bash
python scripts/train_model.py
```

#### **Outputs**:
- Model training accuracy and performance metrics, including:
  - Accuracy: 1.0
  - Precision, Recall, and F1-scores for each class
  - **Feature importance** graph showing which features were most impactful in making predictions.

### **Step 3: Make Predictions**
- The `predict.py` script loads the trained model and uses it to make predictions on new data.

To make predictions:
```bash
python scripts/predict.py
```

#### **Outputs**:
- Predictions for the given dataset.
- Example output:
    ```bash
    Predictions: [1]
    ```

---

## **Results and Insights**

- **Class Distribution after Oversampling**:
    The class distribution is balanced, with each class having an equal number of samples:
    ```plaintext
    Class distribution after oversampling: Counter({0: 12, 1: 12})
    ```

- **Model Evaluation**:
    After training the **Random Forest model**, we obtained perfect accuracy (1.0) on the test set:
    ```plaintext
    Accuracy: 1.0
    ```

    The **classification report** provides more details on the precision, recall, and F1-scores for both classes:
    ```plaintext
    Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      1.00      1.00         2

    accuracy                           1.00         5
    macro avg       1.00      1.00      1.00         5
    weighted avg       1.00      1.00      1.00         5
    ```

    **Confusion Matrix**:
    The confusion matrix visually shows the classification results.

  Feature Importance:
    The feature importance graph shows which features (e.g., GC content, k-mer frequencies) were most influential in the classification model:
    ![Feature Importance](output_images/feature_importance.png)

---

## Conclusions

The project successfully demonstrates how machine learning models, specifically **Random Forest**, can be applied to **biological sequence classification**.
The model has shown excellent performance with an accuracy of 1.0, but future improvements can be made by evaluating it with a larger and more diverse dataset.
The feature extraction process (including GC content and k-mer analysis) plays a significant role in determining sequence characteristics, and the model correctly leverages these features for classification.

---

## Future Work
Data augmentation: Use more diverse biological datasets to improve model generalization.
Model optimization: Tune hyperparameters further and experiment with other models (e.g., SVM, neural networks).
Interpretability: Implement techniques like SHAP or LIME to explain model predictions more transparently.

