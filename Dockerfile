FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app/chatbot_ex.py /code/app/chatbot_ex.py

COPY ./app/main.py /code/app/main.py

COPY ./data /code/data

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
