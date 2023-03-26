FROM python:3.9

WORKDIR /app

COPY ./requirements_light.txt /app/requirements.txt

RUN pip install --upgrade -r /app/requirements.txt

COPY app/main.py app/predict.py app/count_vectorizer.pkl app/linear_regression_model.pkl /app/

RUN pip install uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]