# EquiDefend
import joblib
joblib.dump(best_rf, "human_rights_model.pkl")  # Replace `best_rf` with your trained model
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("human_rights_model.pkl")

class InputData(BaseModel):
    GDP: float
    violation_risk: float
    past_conflict_intensity: float
    region_encoded: int

@app.post("/predict")
async def predict(data: InputData):
    features = np.array([[data.GDP, data.violation_risk, data.past_conflict_intensity, data.region_encoded]])
    prediction = model.predict(features)
    return {"risk_level": int(prediction[0]), "confidence": float(model.predict_proba(features)[0][1])}
    pip install fastapi uvicorn scikit-learn
    uvicorn app:app --reload
    !pip install fastapi nest-asyncio pyngrok uvicorn
from pyngrok import ngrok
import uvicorn
import nest_asyncio

ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
