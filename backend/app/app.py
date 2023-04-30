from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Define the FastAPI app
app = FastAPI()

# Define the input data schema
class InputData(BaseModel):
    input: list

# Load the pre-trained h5 model
model = tf.keras.models.load_model("model.h5")

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert the input data to a numpy array
    input_data = np.array(data.input).reshape((46, 3, 9))

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"prediction": prediction.tolist()}
