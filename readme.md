# Abalone Age Classification Pipeline using K-Nearest Neighbors (KNN)

This project implements a complete machine learning classification pipeline using the classic **Abalone dataset** (sourced from UCI Machine Learning Repository). The goal is to predict the age group (class) of abalones based on their physical measurements and characteristics.

## 🚀 Key Features & Pipeline Steps

1. **Data Ingestion & Column Structure**:
   - Loads the `abalone.data` file without a header, applying standard descriptive column names: `'Sex'`, `'Length'`, `'Diameter'`, `'Height'`, `'Whole_weight'`, `'Shucked_weight'`, `'Viscera_weight'`, `'Shell_weight'`, and `'Rings'`.

2. **Data Hygiene & Inspection**:
   - Checks for missing values (`isnull()`), duplicate rows, and verifies data types and feature distributions.

3. **Categorical Encoding**:
   - Converts the categorical `'Sex'` feature into numerical indicator columns (`'Sex_I'`, `'Sex_M'`, `'Sex_F'`) using one-hot encoding (`pd.get_dummies` with `drop_first=True`).

4. **Target Binning (Multi-class Classification)**:
   - Transforms the continuous target variable `'Rings'` (which indicates age) into 3 distinct classes:
     - **`0`**: Young (`Rings <= 8`)
     - **`1`**: Medium (`8 < Rings <= 11`)
     - **`2`**: Old (`Rings > 11`)

5. **Stratified Dataset Splitting**:
   - Splits data into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets using stratified sampling (`stratify=y`) to maintain class balance across subsets.

6. **Feature Scaling**:
   - Standardizes features using `StandardScaler` fitted exclusively on the training data to prevent data leakage and ensure proper distance-based metric calculations for KNN.

7. **Baseline Model Training**:
   - Trains a baseline `KNeighborsClassifier` ($k=5$) and evaluates performance on the validation set using **Accuracy**.

8. **Hyperparameter Tuning via Grid Search**:
   - Employs `GridSearchCV` with 5-fold cross-validation to find the optimal hyperparameters:
     - **`n_neighbors`**: `[5, 10, 15, 20, 30]`
     - **`metric`**: `['euclidean', 'manhattan']`
     - **`weights`**: `['uniform', 'distance']`

9. **Final Evaluation & Visualization**:
   - Evaluates the best model on the unseen test set using **Accuracy** and **Weighted F1-score**.
   - Generates and plots a **Confusion Matrix** (`ConfusionMatrixDisplay`) to analyze misclassifications between age classes.

## 📦 Requirements

Ensure you have Python installed along with the required data science and machine learning libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## 🛠️ Dataset

This script expects the dataset file **`abalone.data`** to be located in the root directory alongside the script. 

## ▶️ Usage

Place the dataset file in your working directory and execute the script:

```bash
python script.py
```

## 📊 Insights & Observations
The confusion matrix analysis reveals that while the KNN classifier performs reliably on young (`0`) and medium (`1`) abalones, it frequently confuses older abalones (`2`) with medium ones (`1`). This overlap stems from physical growth plateaus in mature abalones, capping overall test accuracy around **65.71%**.