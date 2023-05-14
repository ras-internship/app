from fastapi import FastAPI

import logging
from pydantic import BaseModel
import tensorflow as tf

from functools import reduce
import numpy as np
import os

# Define the FastAPI app
app = FastAPI()

# Create a logger object
logger = logging.getLogger(__name__)

# Load the pre-trained h5 model
model = tf.keras.models.load_model("model.h5")

# Define the input data schema
class InputData(BaseModel):
    input: list

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert the input data to a numpy array
    env_shape = os.environ.get('MODEL_SHAPE', '')
    if env_shape == '':
        env_shape = '3,4'

    pre_shape = list( int(s.strip()) for s in env_shape.split(',') )
    logger.warn(pre_shape)
    y = len(data.input) // reduce(lambda x, y: x * y, pre_shape)

    try:
        input_data = np.array(data.input).reshape((y, *pre_shape))
    except:
        return {"error": "Invalid input shape"}

    try:
        prediction = model.predict(input_data)
    except:
        return {"error": "Invalid predictions"}

    # Return the prediction
    return {"prediction": prediction.tolist()}
