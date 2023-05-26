FROM python:3.10
WORKDIR /ser
COPY ./requirements.txt /ser
RUN pip install pip --no-cache-dir --upgrade -r /ser/requirements.txt
COPY ./ser /serexit
CMD python -c "print(f'Training completed')"
