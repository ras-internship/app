import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

print('Загрузка данных...')
df = pd.read_csv('data_VVP.csv')
print('Размерность данных:', df.shape)
x_test = (df-df.min())/(df.max()-df.min())

time_steps = 3
features = 4

# Инициализация массива x_test
x_test2 = np.zeros((x_test.shape[0] - time_steps + 1, time_steps, features))
for i in range(x_test.shape[0] - time_steps + 1):
    x_test2[i] = x_test.iloc[i:i+time_steps].values
print('Размерность x_test2', x_test2.shape)


# Запрос данных от модели
url = 'http://51.250.76.65:8080/predict'
# url = 'http://127.0.0.1:8000/predict'
headers = {'Content-Type': 'application/json'}
data = {'input': x_test2.flatten().tolist()}
response = requests.post(url, headers=headers, json=data)
if response.ok:
    print('The data sent successfully!')
    resp = response.json()
    vvp = np.array(resp['prediction']) *12000
    plt.plot(vvp)
    plt.grid(True)
    plt.show()
else:
    print('Failed to send the data:', response.status_code)