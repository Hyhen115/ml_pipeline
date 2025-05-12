# Base image with Python
FROM python:3.11-slim

# Install dependencies, including Java
# RUN apt-get update && \
#     apt-get install -y openjdk-17-jdk && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Install dependencies, including Java - downgrade to Java 11 for EMR compatibility
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Java environment
# ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
# ENV PATH=$PATH:$JAVA_HOME/bin

# Set up Java environment for x86_64 architecture
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Verify Java installation
RUN java -version

# Set up Python/PySpark environment
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5001

# Set the environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_RUN_PORT=5001

# Command to run the Flask app
CMD ["python", "app.py"]