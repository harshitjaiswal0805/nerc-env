FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "-c", "import uvicorn; import sys; sys.path.insert(0, '/app'); from server import app; uvicorn.run(app, host='0.0.0.0', port=7860)"]
