
from typing import Tuple, NewType, Union, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# In general, want to use *like types (e.g. Pathlike) at *input* to functions, and use
# the corresponding type without the 'like' suffix for returned values, in accordance
# with the "robustness principle"

Date = pd.Timestamp
# TODO a date-like type, for anything that would be suitable input for pd.Timestamp
# (str, datetime.datetime, pd.Timestamp)?
Datelike = Union[str, datetime, pd.Timestamp]

# TODO update str to some kind of pathlike thing
# TODO rename to Pathlike? just might want some stuff guaranteed to be str (or in
# future, Paths) and some stuff more lenient (like on inputs...)
Pathlike = Union[str, Path]

# TODO see whether specifying an explicit type for this causes problems...
FlyNum = NewType('FlyNum', int)

DateAndFlyNum = Tuple[Date, FlyNum]
PathlikePair = Tuple[Pathlike, Pathlike]
PathPair = Tuple[Path, Path]

# just until i have a proper class/dataclass for them...
OdorDict = NewType('OdorDict', dict)

# TODO are Sequence types guaranteed to work w/ len()? if not, maybe use union of
# list/tuple or something
SingleTrialOdors = Sequence[OdorDict]
# Of length equal to the number of trials in the experiment
ExperimentOdors = Sequence[SingleTrialOdors]

DataFrameOrSeries = Union[pd.DataFrame, pd.Series]
# TODO also include numpy.ndarrays? xr.datasets (some subclass relationship to dataarray
# / better check for both?)? maybe in another type?
# TODO i once was wrapper the args to Union with Type[...]. did that serve a purpose?
DataFrameOrDataArray = Union[pd.DataFrame, xr.DataArray]
NumpyOrXArray = Union[np.ndarray, xr.DataArray]
