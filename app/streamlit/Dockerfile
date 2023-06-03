FROM python:3.10

COPY ./app/streamlit/requirements.txt /st/
COPY ./ml/requirements.txt /ml/

RUN pip install --no-cache-dir --upgrade -r /ml/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /st/requirements.txt

WORKDIR /app

COPY ./ml/ser /app/ml/ser
COPY ./app/streamlit /app

CMD ["streamlit", "run", "--server.address", "0.0.0.0", "--server.port", "8080", "streamlit_ui.py"]
