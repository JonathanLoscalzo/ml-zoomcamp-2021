FROM python:3.9

WORKDIR /code

COPY poetry.lock .
COPY pyproject.toml .


RUN pip install poetry
RUN poetry install

COPY . .

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]