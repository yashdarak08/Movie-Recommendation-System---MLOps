# Use an official Python image as the base
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY src/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the source code
COPY src/ ./src

# Expose port if needed (for inference API)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "src/main.py", "--mode", "inference"]
