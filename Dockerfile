FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask

# Copy all application files
COPY core/ ./core/
COPY tests/ ./tests/
COPY examples_real_world.py .
COPY web_app.py .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs

# Default command
CMD ["python", "web_app.py"]
