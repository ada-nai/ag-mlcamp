FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

RUN pip3 install --upgrade pip

# 2.4.4 2.5.2 2.6.2 2.7.0
RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl?raw=true --no-cache-dir

RUN pip3 install numpy pillow

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
