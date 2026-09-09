FROM python:3.12.13-slim-bookworm

LABEL org.opencontainers.image.title="Spatial-support continuity audit workflow" \
      org.opencontainers.image.description="Containerized environment for the manuscript workflow and companion toolkit" \
      org.opencontainers.image.source="https://github.com/GinixStudy/spatial-support-continuity-audit" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/spatial-support-continuity-audit

COPY requirements_core.txt requirements_full.txt ./
RUN python -m pip install --no-cache-dir -r requirements_full.txt

COPY . .
RUN python -m pip install --no-cache-dir --no-deps . \
    && mkdir -p /kaggle/input/datasets/nanjide/20260711gbif \
                 /kaggle/working/fia_temporal_observation_drift \
                 /kaggle/working/submission_critical_repair_audit \
                 /kaggle/working/recovered_q1

CMD ["python", "container/validate_container.py"]
