"""
Script to train the machine learning model and evaluate performance on slices.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    performance_on_categorical_slice
)
import pickle
import os


# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv('data/census.csv')

# Remove spaces from column names
data.columns = data.columns.str.strip()

# Train-test split
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
print("Processing training data...")
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Train the model
print("Training model...")
model = train_model(X_train, y_train)

# Save the model and encoder
print("Saving model and encoder...")
save_model(model, "model/model.pkl")
with open("model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

# Process the test data with the trained encoder
print("Processing test data...")
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Run inference on test data
print("Running inference...")
preds = inference(model, X_test)

# Compute overall metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"\nOverall Model Performance:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-beta: {fbeta:.4f}")

# Compute performance on slices
print("\nComputing performance on categorical slices...")
slice_feature = "education"
slice_results = performance_on_categorical_slice(
    test,
    slice_feature,
    y_test,
    preds
)

# Write slice output to file
print(f"Writing slice performance to slice_output.txt...")
with open("slice_output.txt", "w") as f:
    f.write(f"Model Performance on {slice_feature} Slices\n")
    f.write("=" * 80 + "\n\n")

    for value, metrics in sorted(slice_results.items()):
        f.write(f"{slice_feature}: {value}\n")
        f.write(f"  Count: {metrics['count']}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F-beta: {metrics['fbeta']:.4f}\n")
        f.write("\n")

print("\nTraining complete!")
print("Model saved to: model/model.pkl")
print("Encoder saved to: model/encoder.pkl")
print("Slice output saved to: slice_output.txt")
