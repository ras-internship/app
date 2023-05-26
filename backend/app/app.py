from fastapi import FastAPI, HTTPException,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import io

import base64
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
)

# Create a logger object
logger = logging.getLogger(__name__)

model_vvp = tf.keras.models.load_model("model_vvp.h5")
model_inf = tf.keras.models.load_model("model_inf.h5")

window_size = 12
start_ahead = 5
steps_ahead = 6

def prediction(model, pdX1):
    Xin = pdX1.iloc[:, 1:]
    scalerX = StandardScaler()
    scaled_dataX = scalerX.fit_transform(Xin)

    scalerY = StandardScaler()
    scalerY.mean_ = scalerX.mean_[0].reshape(-1, 1)
    scalerY.var_ = scalerX.var_[0].reshape(-1, 1)
    scalerY.scale_ = scalerX.scale_[0].reshape(-1, 1)

    X_data = []
    for i in range(len(scaled_dataX) - window_size ):
        X_data.append(scaled_dataX[i:i + window_size])
    npdata = np.array(X_data)

    predictions = []
    for d in npdata:
        input_data = d[np.newaxis, ...]
        prediction = model.predict(input_data, verbose=0)
        sc = scalerY.inverse_transform(prediction).flatten()[0]
        predictions.append(sc.flatten()[0])
    return predictions

def gen_plot(timestamps, predictions, pdX1, title):
    _, (ax1) = plt.subplots(1, 1, figsize=(15, 10), sharex=True)
    Y = pdX1.iloc[:, 1]
    ax1.plot(timestamps[-len(predictions):], predictions, label='Прогн. Y1 +6мес')
    ax1.plot(timestamps[:len(Y)], Y, label='Фактические значения Y')
    ax1.axvline(x=timestamps[len(Y)-1], color='red', linestyle='--', label=f'Граница трениров. данных ({timestamps[len(Y)-1].strftime("%d.%m.%Y")})')
    ax1.set_title(f'Модель для "{title}". Окно={window_size}мес. Прогноз={steps_ahead} мес.')
    ax1.legend()
    ax1.set_ylabel(f"{title}")

    plt.xlabel('Дата')
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    ax1.grid(which='minor', linestyle=':', linewidth='0.5')
    plt.grid()

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Encode the image data as base64
    image_stream.seek(0)
    return base64.b64encode(image_stream.getvalue()).decode("utf-8")

def parse_date(date_string):
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    raise ValueError("Invalid date format: " + date_string)

async def predict(file, model, title, title_ru):
    contents = await file.read()
    file_object = io.BytesIO(contents)

    pdX1 = pd.read_csv(file_object)
    try:
        predictions = prediction(model, pdX1)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid input shape for {title}")

    timestamps = [parse_date(d) for d in pdX1.iloc[:-1, 0]]
    for i in range(steps_ahead):
        timestamps.append(timestamps[-1] + relativedelta(months=+1))

    image_data = gen_plot(timestamps, predictions, pdX1, title_ru)
    return {
#        'predictions': predictions,
        'graph': image_data
    }

@app.post("/vvp")
async def predict_vvp(file: UploadFile = File(...)):
    return await predict(file, model_vvp, 'VVP', 'ВВП')

@app.post("/inf")
async def predict_inf(file: UploadFile = File(...)):
    return await predict(file, model_inf, 'inflation', 'Инфляция')