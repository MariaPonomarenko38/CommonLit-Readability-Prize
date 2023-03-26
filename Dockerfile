FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements_light.txt /app/requirements.txt

RUN pip install --upgrade -r /app/requirements.txt

COPY ./app/main.py ./app/count_vectorizer.pkl ./app/linear_regression_model.pkl /app/app/