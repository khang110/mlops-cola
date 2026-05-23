FROM public.ecr.aws/lambda/python:3.11

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV TRANSFORMERS_CACHE=${LAMBDA_TASK_ROOT}/transformers_cache

COPY requirements_inference.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements_inference.txt -t ${LAMBDA_TASK_ROOT}

COPY . ${LAMBDA_TASK_ROOT}

# Downgrade ONNX model IR version: exported with IR 10, onnxruntime 1.16.3 supports max IR 9
RUN pip install --no-cache-dir onnx && \
    python -c "import onnx; m=onnx.load('${LAMBDA_TASK_ROOT}/models/model.onnx'); m.ir_version=8; onnx.save(m, '${LAMBDA_TASK_ROOT}/models/model.onnx')"

# Pre-download tokenizer so Lambda cold start needs no network access
RUN PYTHONPATH=${LAMBDA_TASK_ROOT} python -c \
    "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')"

CMD ["app.handler"]
