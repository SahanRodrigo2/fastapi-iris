from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Dict

# Load model at startup
bundle = joblib.load("model.pkl")
model = bundle["model"]
target_names = bundle["target_names"]
features = bundle["features"]
accuracy = bundle["accuracy"]

app = FastAPI(title="Iris Classification API", description="Predict iris species", version="1.0")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisInput):
    try:
        # Convert to numpy
        features_arr = np.array([[input_data.sepal_length,
                                  input_data.sepal_width,
                                  input_data.petal_length,
                                  input_data.petal_width]])
        # Predict
        probs = model.predict_proba(features_arr)[0]
        pred_index = int(np.argmax(probs))
        pred_class = target_names[pred_index]
        return PredictionOutput(
            prediction=pred_class,
            confidence=float(probs[pred_index]),
            probabilities={target_names[i]: float(probs[i]) for i in range(len(target_names))}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "LogisticRegression with StandardScaler",
        "problem_type": "classification",
        "features": features,
        "accuracy": accuracy
    }
