FROM python:3.10
LABEL authors="filip"

# Workspace prep
RUN apt-get update
WORKDIR /app

# Copy required files
COPY ./lab ./lab
COPY ./config ./config
COPY ./requirements.txt .

RUN pip install -r requirements.txt

# Expose MLflow UI port
#EXPOSE 5000

CMD ["python", "-m", "lab.experiments.train_agent"]
