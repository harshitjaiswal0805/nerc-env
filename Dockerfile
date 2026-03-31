FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# force rebuild v2
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]# rebuild Tue Mar 31 15:43:33 IST 2026
