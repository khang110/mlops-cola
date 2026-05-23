FROM public.ecr.aws/lambda/python:3.11

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

COPY requirements_inference.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements_inference.txt -t ${LAMBDA_TASK_ROOT}

COPY . ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]