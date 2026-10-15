import subprocess
from pathlib import Path


def test_no_tracked_symlink_leaves_the_repository():
    """A worktree symlinked data/cache to a local path and `git add -A` committed it (caught by review on 2026-10-15).
    A tracked symlink must be relative and stay inside the repo."""
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(["git", "ls-files", "-s"], cwd=root, capture_output=True, text=True, check=True).stdout
    bad = []
    for line in out.splitlines():
        mode, _, rest = line.partition(" ")
        if mode != "120000":
            continue
        path = rest.split("\t", 1)[1]
        target = (root / path).readlink() if (root / path).is_symlink() else None
        if target is None or target.is_absolute() or not (root / path).parent.joinpath(target).resolve().is_relative_to(root.resolve()):
            bad.append(path)
    assert not bad, f"tracked symlinks that point outside the repo: {bad}"
