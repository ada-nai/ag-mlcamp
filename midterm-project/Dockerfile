FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "model.bin", "dv.bin", "predict.py","./"]

RUN pipenv install --system --deploy

EXPOSE 5501

# Below line to be included if deploying to Heroku
ENTRYPOINT [ "gunicorn", "predict:app"]

# Below bind parameter to be included if deploying to localhost
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5501", "predict:app"]
