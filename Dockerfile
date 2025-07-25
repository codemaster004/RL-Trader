FROM python:3.10
LABEL authors="filip"

# Workspace prep
RUN apt-get update
WORKDIR /app

# Install Requirements
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY ./lab ./lab
COPY ./config ./config

# Expose MLflow UI port
#EXPOSE 5000

CMD ["python", "-m", "lab.experiments.train_agent"]
