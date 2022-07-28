
import pytest
from numpy.random import default_rng, Generator


# Adapting strategy proposed in:
# https://github.com/pytest-dev/pytest/issues/667#issuecomment-112206152
# to use more modern, non-global numpy rng.
#
# Also considered using: pytest-randomly or pytest-rng, but it seemed former was doing
# too many things I didn't want by default, and latter wasn't super well supported and
# seemed to change the pytest behavior in ways I'm not sure I want to make sense of at
# the moment.
@pytest.fixture
def rng() -> Generator:
    return default_rng(0)
