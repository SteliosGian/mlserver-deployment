FROM python:3.10.11-slim-buster

WORKDIR /app/

ENV _PIPENV_VERSION 2023.4.29
ENV PIP_NO_CACHE_DIR 1
ENV PIPENV_INSTALL_TIMEOUT 3000

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pipenv==${_PIPENV_VERSION}

COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install --system --deploy --verbose --clear

COPY src .

EXPOSE 8080

CMD ["mlserver", "start", "inference/"]
