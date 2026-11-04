from importlib.util import find_spec


# TODO maybe move to errors-only module / thor?
class TooManyStimulusFiles(Exception):
    pass

# Not specified. Different from specified-but-not-existing-on-disk, which should be an
# IOError
class NoStimulusFile(Exception):
    pass


def require_optional_deps(module_name: str, *import_names: str) -> None:
    """Raises ModuleNotFoundError (suggesting `hong2p[all]`) if any aren't installed.

    For use at the top of hong2p modules that require optional dependencies, before
    importing them.

    Args:
        module_name: name of the hong2p module requiring the dependencies (`__name__`)
        *import_names: top-level names the dependencies are imported by (e.g.
            'matplotlib', not 'matplotlib.pyplot')
    """
    missing = [x for x in import_names if find_spec(x) is None]
    if len(missing) > 0:
        raise ModuleNotFoundError(f'{module_name} requires optional dependencies that '
            f"are not installed ({', '.join(missing)}). Install with: "
            "pip install 'hong2p[all]'", name=missing[0]
        )
