FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml ./
COPY src/ ./src/

RUN uv sync --no-dev

COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src

VOLUME ["/app/data"]

ENTRYPOINT ["uv", "run", "mindly"]
CMD ["--help"]
