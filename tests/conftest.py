"""tests/conftest.py -- suite-wide fixtures.

The live-upstream guard lives here rather than in the one test file that needs
it, so that every test in the suite inherits it: the network-fetching leaves
under `api.routes` are the quota boundary, and a guard scoped to a single file
leaves the next file that reaches one unprotected and saying nothing. The
`build_snapshot`-specific half stays in
`tests/test_snapshot_shape_reconciliation.py`, because the four live builders are
real code under test elsewhere in this suite. See `live_upstream.py` for both
layers and for why the leaves are proxied rather than patched.
"""

from live_upstream import _no_live_upstream  # noqa: F401  (defines the autouse fixture)
