FROM python:3.9.16-slim-buster

WORKDIR /app/

ENV _PIPENV_VERSION 2023.4.20
ENV PIP_NO_CACHE_DIR 1
ENV PIPENV_INSTALL_TIMEOUT 3000

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pipenv==${_PIPENV_VERSION}

COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install --system --deploy --verbose --clear
