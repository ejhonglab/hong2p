
from datetime import datetime
from pathlib import Path
from typing import (Any, Dict, Hashable, Literal, NewType, Optional, Sequence, Tuple,
    Union
)

import numpy.typing as npt
# TODO replace usage of np w/ npt?
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Colormap


# In general, want to use *like types (e.g. Pathlike) at *input* to functions, and use
# the corresponding type without the 'like' suffix for returned values, in accordance
# with the "robustness principle"

# TODO use this in more places
KwargDict = Dict[str, Any]
# TODO rename KwargDict ParamDict and delete this? much al_analysis (and probably
# natmix_data) code already references ParamDict
ParamDict = KwargDict
# TODO use this (/delete. may like the more explicit Optional[KwargDict]...)
OptKwargDict = Optional[Dict[str, Any]]

# TODO rename this and all *like types to *Like, to be consistent w/ np ArrayLike?
# TODO TODO actually use in the many placese i need this
Floatlike = Union[int, float, np.integer, np.floating]

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

# TODO this Literal behaving as expected?
# TODO pandas have some type for this already? mainly want to use this for axis= kwarg
# arguments to pandas fns (when my own fns expose the axis kwarg)
# TODO broaden to also work w/ DataArray indices, or not worth it?
Axis = Literal['columns', 'index', 'rows', 0, 1]

Figsize = Optional[Tuple[float, float]]

# TODO use in viz?
# TODO matplotlib.typing was just introduced in 3.8.0, and i'm on 3.7.3. if i'm able to
# upgrade, def Color from mpl.typing.ColorType
# copied from def in matplotlib documentation
Color = Union[
    str, Tuple[float, float, float], Tuple[float, float, float, float],
    # TODO what are these last two for?
    Tuple[Tuple[float, float, float], str, float],
    Tuple[Tuple[float, float, float, float], float]
]

# TODO use in viz?
# TODO also something in mpl >=3.8 typing for this?
CMap = Union[str, Colormap]

# TODO use more places
Palette = Union[CMap, Dict[Hashable, Color]]

MplRotation = Union[Floatlike, Literal['vertical', 'horizontal']]
