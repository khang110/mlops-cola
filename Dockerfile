FROM public.ecr.aws/lambda/python:3.11

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

COPY ./ ${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install pysqlite3-binary
RUN pip install -r requirements_inference.txt

RUN python3 -c "import site,os; sc=os.path.join(site.getsitepackages()[0],'sitecustomize.py'); open(sc,'w').write('import sys\ntry:\n    import pysqlite3\n    sys.modules[\"sqlite3\"]=pysqlite3\nexcept Exception:\n    pass\n')"

CMD ["app.handler"]