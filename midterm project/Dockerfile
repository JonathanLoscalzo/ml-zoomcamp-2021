FROM python:3.9

WORKDIR /code

COPY poetry.lock .
COPY pyproject.toml .

RUN pip install -U pip 
RUN pip install poetry
ENV PATH="${PATH}:/root/.poetry/bin"

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

COPY . .

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]