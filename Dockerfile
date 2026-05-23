FROM python:3.11-slim
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

COPY ./ /app
WORKDIR /app

# install requirements
RUN pip install "dvc[s3]"
RUN pip install -r requirements_inference.txt

# initialise dvc
# RUN dvc init
# configuring remote server in dvc
RUN dvc remote add -d -f model-store s3://amz-models-dvc/trained_models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]