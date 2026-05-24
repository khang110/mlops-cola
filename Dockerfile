FROM public.ecr.aws/lambda/python:3.11

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

COPY ./ ${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install "dvc[s3]" pysqlite3-binary
RUN pip install -r requirements_inference.txt

RUN python3 -c "import site,os; sc=os.path.join(site.getsitepackages()[0],'sitecustomize.py'); open(sc,'w').write('import sys\ntry:\n    import pysqlite3\n    sys.modules[\"sqlite3\"]=pysqlite3\nexcept Exception:\n    pass\n')"

RUN dvc remote add -d -f model-store s3://amz-models-dvc/trained_models/

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN
ARG AWS_DEFAULT_REGION=us-east-1

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

RUN dvc pull dvcfiles/trained_model.dvc

CMD ["app.handler"]