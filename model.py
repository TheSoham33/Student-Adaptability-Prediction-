import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset (LOCAL PATH)
fp = "students_adaptability_level_online_education_train.csv"
df = pd.read_csv(fp)

# Dataset info
df.info(), df.head()

# Check missing values
missing_values = df.isnull().sum()
print(missing_values)

# Unique values in each column
unique_values = df.nunique()
print(unique_values)

# Set plot style
sns.set(style="whitegrid")
my_colors = ["skyblue", "lightcoral", "lightgreen", "lightyellow", "lightpink", "lightcyan"]

# Plot categorical distributions
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    if df[col].nunique() <= 6:
        sns.countplot(y=df[col], ax=axes[i], order=df[col].value_counts().index, palette=my_colors[:df[col].nunique()])
        axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# Mapping age and class duration
age_mapping = {"06-Oct": "6-10", "01-May": "1-5", "Nov-15": "11-15"}
df["Age"] = df["Age"].replace(age_mapping)

class_duration_mapping = {"03-Jun": "3-6", "01-Mar": "1-3", "0": "0"}
df["Class Duration"] = df["Class Duration"].replace(class_duration_mapping)

# Select categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
columns_to_remove = ['Gender', 'Institution Type', 'IT Student', 'Location', 'Load-shedding', 'Internet Type', 'Self Lms']
categorical_columns = [col for col in categorical_columns if col not in columns_to_remove]

# One-hot encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_data = encoder.fit_transform(df[columns_to_remove])
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(columns_to_remove))

encoder = OneHotEncoder(sparse_output=False)
categorical_data = encoder.fit_transform(df[categorical_columns])
categorical_df_ = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Merge encoded data
merged_df = pd.concat([categorical_df, categorical_df_], axis=1)

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the dataset
file_path = "categorical_data.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Combine one-hot encoded labels into a single target column
df["Adaptivity Level"] = df[["Adaptivity Level_High", "Adaptivity Level_Low", "Adaptivity Level_Moderate"]].idxmax(axis=1)

# Drop the original one-hot encoded columns
df.drop(["Adaptivity Level_High", "Adaptivity Level_Low", "Adaptivity Level_Moderate"], axis=1, inplace=True)

# Separate features and target variable
X = df.drop(columns=["Adaptivity Level"])
y = df["Adaptivity Level"]

# Print class distribution before balancing
print("Class distribution before balancing:", Counter(y))

# Apply SMOTE for oversampling the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["Adaptivity Level"])], axis=1)

# Convert back to one-hot encoding with 1s and 0s
df_balanced = pd.get_dummies(df_balanced, columns=["Adaptivity Level"], dtype=int)

df_balanced.rename(columns={
    "Adaptivity Level_Adaptivity Level_High": "Adaptivity Level_High",
    "Adaptivity Level_Adaptivity Level_Low": "Adaptivity Level_Low",
    "Adaptivity Level_Adaptivity Level_Moderate": "Adaptivity Level_Moderate"
}, inplace=True)

df_balanced = df_balanced.applymap(lambda x: 1 if x > 0 else 0)

# Print class distribution after balancing
print("Class distribution after balancing:", Counter(y_resampled))

# Print class distribution after balancing
print("Class distribution after balancing:", Counter(y_resampled))

# Save the balanced dataset
df_balanced.to_csv("balanced_data.csv", index=False)
print("Balanced dataset saved successfully.")


# Define target
target = ["Adaptivity Level_Low", "Adaptivity Level_Moderate", "Adaptivity Level_High"]
X = df_balanced.drop(columns=target)
y = df_balanced[target].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
try:
    xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method='hist')
    xgb_model.fit(X_train, y_train)
except Exception as e:
    print("Error!!")

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Evaluate model
try:
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
except Exception as e:
    print("Error!!", e)

# Save the model (LOCAL PATH)
joblib.dump(xgb_model, 'model.pkl')
xgb_model = joblib.load('model.pkl')

# Load test dataset (LOCAL PATH)
test_file_path = "students_adaptability_level_online_education_test.csv"
test = pd.read_csv(test_file_path)

# Apply mappings
test["Age"] = test["Age"].replace(age_mapping)
test["Class Duration"] = test["Class Duration"].replace(class_duration_mapping)

# Select categorical columns
test_categorical_columns = test.select_dtypes(include=['object']).columns.tolist()
test_columns_to_remove = ['Gender', 'Institution Type', 'IT Student', 'Location', 'Load-shedding', 'Internet Type', 'Self Lms']
test_categorical_columns = [col for col in test_categorical_columns if col not in test_columns_to_remove]

# One-hot encoding for test data
encoder = OneHotEncoder(drop='first', sparse_output=False)
test_categorical_data = encoder.fit_transform(test[test_columns_to_remove])
test_categorical_df = pd.DataFrame(test_categorical_data, columns=encoder.get_feature_names_out(test_columns_to_remove))

encoder = OneHotEncoder(sparse_output=False)
test_categorical_data = encoder.fit_transform(test[test_categorical_columns])
test_categorical_df_ = pd.DataFrame(test_categorical_data, columns=encoder.get_feature_names_out(test_categorical_columns))

# Merge encoded test data
merged_df = pd.concat([test_categorical_df, test_categorical_df_], axis=1)

# Save preprocessed test data (LOCAL PATH)
merged_df.to_csv("test_categorical_data.csv", index=False)
print("CSV file 'test_categorical_data.csv' has been created successfully.")

# Predict on test set
A = merged_df.drop(columns=target)
predictions = xgb_model.predict(A)
predictions = predictions.round().astype(int)

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions, columns=["Adaptivity Level_Low", "Adaptivity Level_Moderate", "Adaptivity Level_High"])

# Save predictions (LOCAL PATH)
pred_df.to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'.")

