FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy the content of the current directory to the container
COPY . /app

# Install system dependencies and awscli
RUN apt-get update -y && \
    apt-get install -y awscli && \
    pip install --upgrade pip setuptools

# Install Python dependencies
RUN pip install -r requirements.txt

# Run the application
CMD [ "python3", "app.py" ]
