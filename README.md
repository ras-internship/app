# app

You can use this image for any models.

You should mount a volume with the model, and have to set up the shape as an environment variable `MODEL_SHAPE`:
```sh
$ cd backend
$ docker build -t backend:latest .

$ docker run -it --rm -e MODEL_SHAPE="3,46,9" -v /home/amf/ras-internship/model.h5:/app/model.h5 -p 8080:8080 backend:latest
```
