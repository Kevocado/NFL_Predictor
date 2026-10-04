"""quantile_registry.py -- save, load and verify the versioned quantile artifacts.

Separate from `player_props.py` because this is about files on disk and a
manifest, not about fitting. The versioned artifacts sit beside the production
point-regressor pickles under a distinct suffix, so a stale artifact directory
still serves the live site and the two can be compared side by side.

Every artifact records a sha256 in the manifest. The point-regressor models in
this repo already learned that lesson the hard way -- `models/manifest.py`
carries a long note about a fingerprint that was compared against itself and
therefore verified nothing.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from .player_props import QUANTILES

#: Manifest key. Versioned so a future artifact family cannot silently overwrite
#: this one.
ARTIFACT_KEY = "quantile_yardage_v1"

#: Suffix for every file this module writes.
ARTIFACT_SUFFIX = "_quantile_2025"


def artifact_path(market: str, out_dir: Path | str) -> Path:
    return Path(out_dir) / f"{market}{ARTIFACT_SUFFIX}.pkl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_quantile_artifacts(artifacts: dict[str, dict], out_dir: Path | str,
                            manifest: dict, manifest_path: Path | str,
                            quantiles: list[float] | None = None) -> dict:
    """Write one pickle per market and extend the manifest with `ARTIFACT_KEY`.

    The manifest is extended, never rewritten: every existing key is carried
    through untouched, because the live site reads them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markets: dict[str, dict] = {}
    trained_seasons: list[int] | None = None
    feature_cols: list[str] | None = None

    for market, payload in artifacts.items():
        if not payload.get("quantile_models"):
            raise ValueError(f"{market}: artifact carries no 'quantile_models'")
        path = artifact_path(market, out_dir)
        path.write_bytes(pickle.dumps(payload))
        markets[market] = {
            "path": path.name,
            "sha256": _sha256(path),
            "walkforward_mae": payload.get("walkforward_mae"),
            "trained_seasons": payload.get("trained_seasons"),
            "n_features": len(payload.get("feature_cols") or []),
            "n_quantiles": len(payload["quantile_models"]),
        }
        trained_seasons = trained_seasons or payload.get("trained_seasons")
        feature_cols = feature_cols or payload.get("feature_cols")

    extended = dict(manifest)
    extended[ARTIFACT_KEY] = {
        "artifact_suffix": ARTIFACT_SUFFIX,
        "quantiles": list(quantiles or QUANTILES),
        "trained_seasons": trained_seasons,
        "feature_cols": feature_cols,
        "markets": markets,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(manifest_path).write_text(json.dumps(extended, indent=2) + "\n")
    return extended


def load_quantile_artifact(market: str, out_dir: Path | str) -> dict:
    return pickle.loads(artifact_path(market, out_dir).read_bytes())


def verify_quantile_artifacts(manifest: dict, out_dir: Path | str) -> list[str]:
    """Return a list of problems; empty means every artifact matches the manifest.

    A missing `ARTIFACT_KEY` is a problem, not a pass -- the absence of a
    fingerprint is exactly the state this is here to detect.
    """
    entry = manifest.get(ARTIFACT_KEY)
    if not entry:
        return [f"manifest has no {ARTIFACT_KEY!r} entry"]

    problems: list[str] = []
    for market, meta in (entry.get("markets") or {}).items():
        path = Path(out_dir) / meta["path"]
        if not path.exists():
            problems.append(f"{market}: artifact {path.name} is missing")
            continue
        digest = _sha256(path)
        if digest != meta.get("sha256"):
            problems.append(f"{market}: sha256 does not match the manifest "
                            f"(file {digest[:12]}..., manifest {str(meta.get('sha256'))[:12]}...)")
    return problems