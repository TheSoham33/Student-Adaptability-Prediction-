import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# # Define paths
# BASE_DIR = r"D:\Projects\Student Adaptibility"
# train_file_path = os.path.join(BASE_DIR, "students_adaptability_level_online_education_train.csv")
# test_file_path = os.path.join(BASE_DIR, "students_adaptability_level_online_education_test.csv")
# model_path = os.path.join(BASE_DIR, "model.pkl")
# output_path = os.path.join(BASE_DIR, "predictions.csv")


# Define paths - **Using Relative Paths **
train_file_path = "students_adaptability_level_online_education_train.csv"
test_file_path = "students_adaptability_level_online_education_test.csv"
model_path = "model.pkl"
output_path = "predictions.csv"

# Load the training dataset
if not os.path.exists(train_file_path):
    raise FileNotFoundError(f"❌ Training file not found: {train_file_path}")

df = pd.read_csv(train_file_path)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_data = encoder.fit_transform(df[categorical_columns])
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Merge encoded data with numerical features
df = df.drop(columns=categorical_columns).reset_index(drop=True)
df = pd.concat([df, categorical_df], axis=1)

# Define target and features
target = ["Adaptivity Level_Low", "Adaptivity Level_Moderate"]
X = df.drop(columns=target)
y = df[target].astype(int)  # Convert to numerical format

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the model already exists
if os.path.exists(model_path):
    print("✅ Model found. Loading existing model...")
    xgb_model = joblib.load(model_path)
else:
    print("⚠️ Model not found. Training a new model...")

    # Train the XGBoost model
    try:
        xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method='hist')
        xgb_model.fit(X_train, y_train)
        print("✅ Model trained successfully!")
    except Exception as e:
        raise RuntimeError(f"❌ Error during model training: {e}")

    # Save the trained model
    joblib.dump(xgb_model, model_path)
    print(f"📁 Model saved at: {model_path}")

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate model
try:
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"📊 Accuracy: {accuracy:.2f}")
    print("📜 Classification Report:\n", report)
except Exception as e:
    print("❌ Error during evaluation:", e)

# Load the test dataset
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"❌ Test file not found: {test_file_path}")

dt = pd.read_csv(test_file_path)

# Process categorical features in the test dataset
categorical_columns = dt.select_dtypes(include=['object']).columns.tolist()
categorical_data = encoder.transform(dt[categorical_columns])
categorical_dt = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

dt = dt.drop(columns=categorical_columns).reset_index(drop=True)
dt = pd.concat([dt, categorical_dt], axis=1)

# Ensure feature order matches training data
# Ensure feature order matches training data
expected_features = getattr(xgb_model, "feature_names_in_", None)

# Ensure feature order matches training data
expected_features = getattr(xgb_model, "feature_names_in_", None)
if expected_features is not None and len(expected_features) > 0:
    dt = dt.reindex(columns=expected_features, fill_value=0)


# Make predictions on test data
predictions = xgb_model.predict(dt).round().astype(int)

# Save predictions
pred_df = pd.DataFrame(predictions, columns=["Adaptivity Level_Low", "Adaptivity Level_Moderate"])
pred_df.to_csv(output_path, index=False)

print(f"✅ Predictions saved to: {output_path}")
