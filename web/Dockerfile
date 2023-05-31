FROM python:3.10

COPY ./web/requirements.txt /web/
COPY ./ml/requirements.txt /ml/

RUN pip install --no-cache-dir --upgrade -r /ml/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /web/requirements.txt

WORKDIR /app

COPY ./ml/ser /app/ml/ser
COPY ./web/emotions_recognition_api /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
