For now, try to preserve support for Python 3.8 (with last pandas 1.5), as well as the
current ./venv/ python (should be 3.14 with pandas 3). Do this by either
asking-for-forgiveness-not-permission (e.g. try/except handling), or explicit checking
of versions if necessary. If code only exists to support old pandas 1.5 (or other old py
thon 3.8 stuff), add a comment saying which old thing that part of the code is for.
