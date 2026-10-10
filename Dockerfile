FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# Dependencies FIRST, from pyproject.toml alone. This is the ~1.2 GB layer (xgboost, pandas, scipy, scikit-learn).
# It used to sit AFTER `COPY src/`, so every commit invalidated it and CI built a brand-new 1.2 GB layer with a
# new digest each time; the VPS (38 GB) then held a full extra copy per deploy plus one for the rollback, and filled
# to 100% on 2026-10-09. With the dependency install keyed only on pyproject.toml, an unchanged dependency set
# reuses the SAME layer digest across commits and a deploy adds only the thin app layers.
# `pip install -e .` needs the package directory to exist to resolve it, so a stub stands in for the real source;
# the real source is copied right after and PYTHONPATH=/app/src (set below) is what the app imports from.
COPY pyproject.toml ./
RUN mkdir -p src/nfl_predictor && touch src/nfl_predictor/__init__.py \
    && pip install --no-cache-dir -e . \
    && rm -rf src

COPY src/ ./src/
COPY models/ ./models/

# The precomputed games/predictions/player-props this deployment actually
# serves (see public_snapshot.py's module docstring) -- generated locally
# or by .github/workflows/refresh-public-snapshot.yml
# (`python -m nfl_predictor.public_snapshot`) and committed, not built in
# this image. Must exist before building.
COPY data/public_snapshot.json ./data/public_snapshot.json

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app/src
ENV PUBLIC_MODE=true

EXPOSE 8001
CMD ["sh", "-c", "uvicorn nfl_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
