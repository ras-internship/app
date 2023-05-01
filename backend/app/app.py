from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

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
    env_shape = os.environ['MODEL_SHAPE']
    shape = tuple( int(s.strip()) for s in env_shape.split(',') )
    input_data = np.array(data.input).reshape(shape)
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"prediction": prediction.tolist()}
