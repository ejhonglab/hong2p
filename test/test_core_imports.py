"""
Checks modules that should only require the core dependencies (not `hong2p[all]`) do not
import any optional dependencies at import time.

Runs in any environment (each import in a fresh subprocess, since other tests may have
already imported optional dependencies), so this also catches regressions in the full
dev venv. The tox `core` env additionally checks the core dependencies are sufficient.
"""

import subprocess
import sys

import pytest


# NOTE: if changing, also update comment above `dependencies` in pyproject.toml, and
# tox `core` env (doctest) paths
CORE_MODULES = [
    'hong2p',
    'hong2p.err',
    'hong2p.types',
    'hong2p.pandas',
    'hong2p.olf',
    'hong2p.thor',
    'hong2p.util',
    'hong2p.matlab',
    'hong2p.chem',
    'hong2p.cli_entry_points',
]

# Names these are imported by (not distribution names), for everything in `all` extra in
# pyproject.toml
OPTIONAL_DEPS = [
    'importlib_resources',
    'scipy',
    'xarray',
    'netCDF4',
    'PyQt5',
    'pyqtgraph',
    'matplotlib',
    'seaborn',
    'matplotlib_scalebar',
    'colorcet',
    'sklearn',
    'git',
    'jinja2',
    'latex',
    'requests',
    'mol2chemfigPy3',
    'ijroi',
    'termcolor',
    'psutil',
    # (not pyarrow, also in `all`, which pandas >= 3 imports itself, if installed)
]


@pytest.mark.parametrize('module', CORE_MODULES)
def test_core_module_imports_without_optional_deps(module):
    code = (f'import sys; import {module}; '
        f'print(sorted(set({OPTIONAL_DEPS!r}) & set(sys.modules)))'
    )
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert proc.returncode == 0, f'importing {module} failed:\n{proc.stderr}'

    imported_optional_deps = proc.stdout.strip()
    assert imported_optional_deps == '[]', \
        f'importing {module} also imported optional deps: {imported_optional_deps}'
