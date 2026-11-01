# CFPB service-failure data pipeline (Module 3)
# Build:  docker build -t cfpb-pipeline .
# Run:    docker run --rm -e PIPELINE_ENV=prod -e PIPELINE_PSEUDONYM_KEY=<secret> \
#           -v "$PWD/data/raw:/app/data/raw:ro" -v "$PWD/out:/out" cfpb-pipeline \
#           --source data/raw/complaints.csv.zip --root /out
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PREFECT_HOME=/tmp/prefect \
    PIPELINE_ENV=prod

WORKDIR /app

# Dependencies first so the layer is cached until the pins change
COPY requirements-pipeline.txt .
RUN pip install -r requirements-pipeline.txt

COPY src ./src
COPY flows ./flows
COPY reference ./reference
COPY governance ./governance
COPY tests ./tests
COPY pyproject.toml .

# Run as an unprivileged user; /out is the only writable data location
RUN useradd --create-home --uid 10001 pipeline \
 && mkdir -p /out \
 && chown -R pipeline /app /out
USER pipeline
VOLUME ["/out"]

# Fail the build image check if the pipeline cannot even import
RUN python -c "import flows.ccdb_pipeline"

ENTRYPOINT ["python", "-m", "flows.ccdb_pipeline"]
CMD ["--source", "data/raw/complaints.csv.zip", "--root", "/out"]
