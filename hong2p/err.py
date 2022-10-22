
# TODO maybe move to errors-only module / thor?
class TooManyStimulusFiles(Exception):
    pass

# Not specified. Different from specified-but-not-existing-on-disk, which should be an
# IOError
class NoStimulusFile(Exception):
    pass

