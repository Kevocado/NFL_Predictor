"""The stale marker has to be readable by the browser that asks for it.

Important 4 added `X-Player-Props-Stale` so a `stale` week could not be passed off
as a fresh one, and round 1 shipped it that way: header set, client read it,
component rendered it. None of that fires. `main.py` configures CORSMiddleware
with `allow_origins=["*"]` and no `expose_headers`, and per the Fetch spec a
cross-origin response exposes only the CORS-safelisted response headers to
`Headers.get()`. `X-Player-Props-Stale` is not one of them, so a browser hands the
client `null`, `stale` is permanently `false`, and the notice never renders.

Reproduced against the app's own middleware before the fix:

    status: 200
    raw header present          : true
    Access-Control-Allow-Origin : *
    Access-Control-Expose-Headers: None
    => res.headers.get('X-Player-Props-Stale') = null

This is the same defect as the one round 1 fixed, one layer out: a marker nobody
can see. Three parts had to be right at once -- the server sets it, the server
declares it readable, and the client reads it -- and only the middle one was
missing.

**Why this is tested server-side and not in the DOM runner.** CORS is enforced by
the browser, and the entire decision is made by the server: what a cross-origin
caller may read is exactly what `Access-Control-Expose-Headers` lists. The server
response header *is* the contract, so a request-level test pins it exactly and
cannot be made to pass by anything a DOM shim does. `frontend/props_state_check.mjs`
replaces `fetch` wholesale, so it can never exercise this -- not a gap in it, a
property of it. What it does check is the other two parts: that the client reads
the header at all, and that the component renders when the flag is set.

`test_the_stale_marker_survives_a_same_origin_deployment` is the caveat that cannot
be discharged here, and it is a test rather than a comment because the answer
depends on the deploy, not on this repo.
"""
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api import facts as facts_mod
from nfl_predictor.api import routes
from nfl_predictor.api.main import app
from nfl_predictor import public_snapshot

SEASON = 2026
WEEK = 3
CROSS_ORIGIN = "https://app.example.com"


@pytest.fixture
def stale_week(monkeypatch):
    """PUBLIC_MODE with a week the last build could not refresh."""
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {"season": SEASON, "weeks": {
        str(WEEK): {
            "games": [{"game_id": f"{SEASON}_{WEEK:02d}_BAL_KC"}],
            "predictions": {},
            "player_props_status": "stale",
            "player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                              "position": "RB", "anytime_td_prob": 0.42}],
        }
    }})
    return TestClient(app)


def test_the_stale_header_is_set_on_a_stale_week(stale_week):
    """The part that already worked, pinned so the other two cannot pass without it."""
    response = stale_week.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200
    assert response.headers.get(routes.PROPS_STALE_HEADER) == "true"


def test_the_stale_header_is_exposed_to_cross_origin_callers(stale_week):
    """The part that was missing, and the reason the marker never reached a reader.

    Deleting `expose_headers` from main.py's CORSMiddleware leaves this file green
    except for this test: the header is still set, the client still asks for it,
    the component still renders -- and a browser still sees `null`.
    """
    response = stale_week.get(
        f"/api/players/{SEASON}/{WEEK}/props", headers={"Origin": CROSS_ORIGIN}
    )

    assert response.status_code == 200
    exposed = response.headers.get("access-control-expose-headers") or ""
    assert routes.PROPS_STALE_HEADER.lower() in exposed.lower(), (
        f"{routes.PROPS_STALE_HEADER} is set on the response but not declared readable, so a "
        f"cross-origin browser cannot see it. Access-Control-Expose-Headers: {exposed!r}"
    )


def test_the_marker_header_is_exposed_globally_not_only_on_the_props_route(monkeypatch):
    """Expose-Headers is a per-response header, so the middleware has to know the
    name. A second consumer may add one later, and the failure mode -- a header
    silently arriving as null on whichever route forgot to opt in -- is exactly the
    bug this file exists to prevent. So the middleware's configuration is asserted
    directly, rather than inferred from one route's behaviour."""
    configured = None
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            configured = middleware.kwargs
    assert configured is not None, "CORSMiddleware is not installed; the guard below is vacuous"
    assert routes.PROPS_STALE_HEADER in (configured.get("expose_headers") or []), (
        f"expose_headers is {configured.get('expose_headers')!r}, which does not include "
        f"{routes.PROPS_STALE_HEADER!r}"
    )


def test_a_fresh_week_declares_the_same_readable_set(stale_week, monkeypatch):
    """Expose-Headers is emitted on every response, so it must not depend on the
    route having produced a marker. A reader switching between a fresh week and a
    stale one must not see the header list change under them."""
    from nfl_predictor.api import routes as r
    monkeypatch.setattr(r, "_public_snapshot_cache", {"season": SEASON, "weeks": {
        str(WEEK): {"games": [{"game_id": "g"}], "predictions": {}, "player_props_status": "ok",
                   "player_props": [{"player_id": "p1", "player_name": "A. Back",
                                     "recent_team": "BAL", "position": "RB",
                                     "anytime_td_prob": 0.42}]}
    }})

    response = TestClient(app).get(
        f"/api/players/{SEASON}/{WEEK}/props", headers={"Origin": CROSS_ORIGIN}
    )

    exposed = (response.headers.get("access-control-expose-headers") or "").lower()
    assert r.PROPS_STALE_HEADER.lower() in exposed
    assert r.PROPS_STALE_HEADER not in response.headers, (
        "a fresh week must not carry the marker itself -- only the declaration that "
        "makes the marker readable is unconditional"
    )


def test_the_frontend_base_url_is_cross_origin_in_a_deployed_build():
    """**The caveat this repo cannot discharge, stated as a test.**

    Everything above only matters if the browser really is cross-origin. Evidence
    that it is: `main.py` never mounts `frontend/dist` (there is no `StaticFiles`
    anywhere under `src/`), the Dockerfile copies `dist` into the image without
    serving it, and the container exposes only uvicorn on 8001. So the API has no
    same-origin route to the bundle and the frontend must be hosted elsewhere.

    If a gateway ever puts the bundle and the API behind one origin, the whole
    marker path is inert -- the header would be same-origin and always readable --
    and nothing in this file would notice. That is a deployment change, not a code
    change, so it is recorded here and flagged in the fix report for whoever
    deploys it. This test asserts the *current* arrangement so the day it stops
    being true, it fails loudly instead of silently becoming pointless.
    """
    from nfl_predictor.api import main as main_mod

    mounted = [
        route for route in main_mod.app.routes
        if route.__class__.__name__ == "Mount" and "static" in str(getattr(route, "path", "")).lower()
    ]
    assert not mounted, (
        "the API now serves frontend/dist itself, so the frontend is same-origin and the "
        "X-Player-Props-Stale marker path is inert. Re-evaluate: if it is same-origin the "
        "expose_headers fix is harmless but unnecessary, and the marker still works."
    )


def test_the_public_503_body_uses_the_same_unit_as_the_frontend_copy():
    """The backend sentence is rendered verbatim by `client.ts` and the component
    (`PlayerPropsPage.tsx` puts `err.message` in the alert), so "for this game" on a
    week-keyed route was the same unit mismatch as the empty state said it was not.
    The frontend runner is what caught this, and it caught it by feeding its own
    wording into the 503 case -- so this test pins the backend string to the unit
    the route is keyed on, and the runner is pinned to the backend string."""
    detail = routes.PROPS_UNAVAILABLE_DETAIL
    assert "this week" in detail, f"the 503 body names the wrong unit: {detail!r}"
    assert "this game" not in detail, f"the 503 body names the wrong unit: {detail!r}"
    assert "unavailable" in detail
