FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY conda.yaml .

RUN conda env create -f conda.yaml

ENV PATH=/opt/conda/envs/mlflow-env/bin:$PATH

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "python", "modelling.py"]
