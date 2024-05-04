# Use Python 3.10.9 as the base image
FROM python:3.10.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

CMD ["python", "models/SAC/metra.py"]
