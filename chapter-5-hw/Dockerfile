FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["./hw5_predict.py", "./"]

EXPOSE 5501

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5501", "hw5_predict:app"]
