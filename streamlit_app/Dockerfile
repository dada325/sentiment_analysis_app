# Use a base image with Python
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local directory content into the container's /app directory
COPY . /app/

# Install necessary Python packages
RUN pip install streamlit transformers tensorflow

# Specify the command to run on container start
CMD ["streamlit", "run", "app.py"]
