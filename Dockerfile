FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/

RUN pip install --no-cache-dir -e .

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
