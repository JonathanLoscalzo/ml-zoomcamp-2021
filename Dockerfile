FROM agrigorev/zoomcamp-model:3.8.12-slim

WORKDIR /app

RUN pip install pipenv

COPY ./data/models .
COPY ./notebooks/Pipfile Pipfile
COPY ./notebooks/Pipfile.lock Pipfile.lock

RUN ls
RUN pipenv install --system --deploy

COPY ./scripts/homework5/serving.py .

EXPOSE 1235

ENV FILE_PATH=./

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:1235", "serving:app"]