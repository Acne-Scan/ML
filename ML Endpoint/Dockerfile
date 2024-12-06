# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files to the container
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for the model path
ENV MODEL_PATH=/app/best_model2.tflite

# Set the command to run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_test:app
