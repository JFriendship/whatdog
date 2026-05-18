FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8080

RUN useradd app
USER app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]