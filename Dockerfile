# Use an official Python runtime as a parent image
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir  /usr/src/app/chess_engine

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app/chess_engine

# Update the package list and install neovim
RUN apt-get update && \
    apt-get install -y neovim tmux sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /usr/src/app/chess_engine/pi_requirements.txt

# Define environment variable
# ENV trainModel True
# ENV selfTrain True
# ENV GOOGLE_APPLICATION_CREDENTIALS '/var/secrets/google/key.json'
# ENV BUCKET_NAME "chess-model-weights"
# ENV saveToBucket True

#CMD ["python","./chess_engine/src/model/main.py"]
CMD while true; do sleep 10; done