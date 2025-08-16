from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build pipeline: scaler + logistic regression
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, random_state=42))
])

pipe.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"Test Accuracy: {acc:.3f}")

# Save model and metadata
bundle = {
    "model": pipe,
    "target_names": iris.target_names,
    "features": iris.feature_names,
    "accuracy": acc
}
joblib.dump(bundle, "model.pkl")
print("Model saved to model.pkl")
