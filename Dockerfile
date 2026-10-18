# small official Python image
FROM python:3.12.7-slim

LABEL org.opencontainers.image.source=https://github.com/mameeed/su-maio-group-ac-3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# train the model during the build, so it is baked into the image
COPY train.py app.py ./
RUN python train.py

EXPOSE 80

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
