# app

Docker образ для запуска модели в виде web-сервиса.

You must mount a volume with the two models:
```sh
$ cd backend
$ docker build -t backend:latest .

$ docker run -d --rm \
     -v /home/amf/model_inf.h5:/app/model_inf.h5 \
     -v /home/amf/model_vvp.h5:/app/model_vvp.h5 \
     -p 8080:8080 --name backend backend:latest
```

## Запуск локально
```sh
$ uvicorn app:app --reload
```

## Проверка API через curl
```sh
$ curl -F "file=@vvp.csv" http://localhost:8080/vvp
```
