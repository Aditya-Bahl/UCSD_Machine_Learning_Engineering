# Dockerfile, Image, Container

FROM python:3.9

# Set the working directory to /app and copy current dir
WORKDIR /app
COPY Script1.py /app
COPY requirements.txt /app
COPY config.json /app
COPY sunlit-inquiry-319400-e94abc7798aa.json /app

#ADD script1.py .

ENTRYPOINT [ "python3" ]

RUN pip install -r requirements.txt

CMD ["Script1.py"]