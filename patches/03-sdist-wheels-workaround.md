# Workaround: sdist-only packages failing to build in restricted environments

`uv.lock` records sdist-only sources for: antlr4-python3-runtime, asciitree,
bddl, deepspeed, easydict, egl-probe, future, gym, robomimic, sharedarray,
timeout-decorator, toppra.

On hosts where setuptools' post-build cleanup is intercepted (bulk delete
protection), builds of large packages (bddl=509, future=711, deepspeed=~900
staged files) fail deterministically.

Workaround:
1. Build each sdist manually with `--keep-temp` (cleanup never runs, nothing
   gets blocked):
   `python setup.py bdist_wheel --keep-temp --dist-dir /data/wheels`
2. Pre-install wheels: `uv pip install --no-deps /data/wheels/*.whl`
3. Exclude them from exact sync:
   `uv sync --no-install-package <pkg> ...` for each of the 12 packages
   (order matters: run this BEFORE the final `uv pip install`, since
   `uv sync` removes anything not in its plan).
