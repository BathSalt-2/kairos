FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml README.md LICENSE ./
COPY kairos/ kairos/

# Install
RUN pip install --no-cache-dir ".[all]"

EXPOSE 8000

CMD ["uvicorn", "kairos.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
