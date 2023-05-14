# app

Docker образ для запуска модели в виде web-сервиса.

You should mount a volume with the model, and have to set up the shape as an environment variable `MODEL_SHAPE`:
```sh
$ docker build -t backend:latest .

$ docker run -it --rm -e MODEL_SHAPE="3,4" -v /home/amf/ras-internship/model.h5:/app/model.h5 -p 8080:8080 backend:latest
```
**note**: `MODEL_SHAPE` не содержит общее количество тензоров, т.е. пропускает первый индекс. Для набора данныз (90, 3, 4) `MODEL_SHAPE="3,4"`