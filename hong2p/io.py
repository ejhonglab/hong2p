"""
Writing analysis outputs (CSVs, parquet, pickles, JSON, figures), and reading them back.

Paths written via functions wrapped with `produces_output` (e.g. `to_csv`) and via
`savefig` are recorded for the rest of the process, and writing the same path twice
raises `MultipleSavesPerRunException` (usually a mistake). With `check_outputs_unchanged`
set (e.g. by the -c flag added by `add_check_outputs_unchanged_CLI_flag_and_parse_args`),
an existing output is compared to what would replace it, raising an error if it would
change.

Settings are module attributes, read when the functions are called (e.g. set
`hong2p.io.verbose = True` on the module, not on a name imported from it): `verbose`,
`check_outputs_unchanged`, `prompt_if_changed`, `ignore_existing` (see
`should_ignore_existing`), `save_figs`, `plot_fmt` and `exit_after_saving_fig_containing`
(see `savefig`).
"""

import argparse
from collections import namedtuple
import difflib
from functools import wraps
import filecmp
import inspect
import os
from os.path import getmtime
from pathlib import Path
import pickle
from pprint import pformat
import sys
from tempfile import NamedTemporaryFile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Union

from hong2p.err import require_optional_deps
require_optional_deps(__name__, 'matplotlib')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.exceptions import ImageComparisonFailure

from hong2p import util
from hong2p.olf import solvent_str
from hong2p.pandas import pd_allclose
from hong2p.types import Pathlike, DataFrameOrSeries
# read_json not used here, but available from this module alongside the writers
from hong2p.util import (read_json, to_json as _to_json, warn, format_mtime, fly_cols,
    flyroi_cols
)


# if True, paths of saved figures are printed (and al_analysis / its model code use this
# as their general verbosity flag, e.g. set by `al-analysis -v`)
verbose: bool = False


# TODO factor (/rename) into something that also adds the -v/--verbose flag, and sets
# `verbose` accordingly? (lots of scripts just unconditionally set that true, so
# prob not)
def add_check_outputs_unchanged_CLI_flag_and_parse_args(parser: argparse.ArgumentParser,
    *, check_nonmain_option: bool = False) -> argparse.Namespace:
    """Adds -c/-P (and optionally -C) CLI args, parses args, and sets module settings.

    Sets (in this module):
    - `check_outputs_unchanged` (True/False, or 'nonmain' if -C passed)
    - `check_nonmain_outputs_unchanged` (only if `check_nonmain_option=True`)
    - `prompt_if_changed`

    Args:
        check_nonmain_option: if True, also adds -C/--check-nonmain-outputs-unchanged
            (mutually exclusive w/ -c), which only checks outputs not saved under
            `main()`

    Returns args as from `parser.parse_args()`

    Will raise `argparse.ArgumentError` if trying to add any args already defined.
    """
    global check_outputs_unchanged
    global check_nonmain_outputs_unchanged
    global prompt_if_changed

    if check_nonmain_option:
        add_to = parser.add_mutually_exclusive_group()
    else:
        add_to = parser

    # TODO option to warn but not err as well?
    # TODO warn in cases like sensitivity_analysis's deletion of it's root output folder
    # before starting (invalidating these checks...) (err if any folder would be
    # deleted when we have this flag?)
    # TODO maybe this should prompt for pickles/csvs by default (w/ option to
    # approve single or all?)? maybe backup ones that would be replaced too?
    add_to.add_argument('-c', '--check-outputs-unchanged', action='store_true',
        # TODO update doc? is it actually true there are any plot formats i don't
        # support? or at least, this isn't the reason anymore, right? now it should just
        # be anything that mpl fn i'm using (which converts things to png i think) works
        # w/?
        # TODO specifically call out which formats this will/won't work for (png?)
        # work for PDF? implement some kind of image based diffing to support those?
        # TODO or maybe just err if this is passed with an unsupported plot format being
        # requested
        # TODO doc whether pickles (or other outputs saved via produces_output wrapper)
        # are checked for exact file equality (i assume that is the only default
        # implementation)
        help='For CSVs/pickles/plots (saved with my to_[csv|pickle]/savefig wrappers, '
        'or anything save function wrapped by `@produces_output` decorator), check new '
        'outputs against any existing outputs they would overwrite. If there is a '
        'discrepancy, exit with an error. Currently do not support certain plot '
        'formats (where metadata includes things like file creation time, so same '
        'strategy can not be used to check files for equality).'
    )
    flags_added = ['-c']
    if check_nonmain_option:
        add_to.add_argument('-C', '--check-nonmain-outputs-unchanged',
            action='store_true', help='Like -c, but excludes outputs saved in main() '
            'from checks, so that per-panel analysis outputs can be checked separately.'
            ' Outputs that would trigger a warning with this flag will not be '
            'overwritten.'
        )
        flags_added.append('-C')

    flags_str = '/'.join(flags_added)
    parser.add_argument('-P', '--prompt-if-changed', action='store_true',
        help=f'If {flags_str} would trigger an error because a file changed, will '
        'instead prompt about the would-be change, and pause execution until user '
        'indicates whether the file should be overwritten.'
    )
    args = parser.parse_args()

    check_outputs_unchanged = args.check_outputs_unchanged

    if check_nonmain_option:
        check_nonmain_outputs_unchanged = args.check_nonmain_outputs_unchanged
        if check_nonmain_outputs_unchanged:
            assert not check_outputs_unchanged
            check_outputs_unchanged = 'nonmain'

    prompt_if_changed = args.prompt_if_changed
    if prompt_if_changed:
        # TODO raise an argument error in arg parsing instead?
        assert check_outputs_unchanged != False, f'-P can only be passed if {flags_str}'

    return args


# TODO doc what this does in a comment (are any outputs actually written, or all just
# written to temp paths and compared to any existing current outputs at same paths?)
# True|False|'nonmain'
check_outputs_unchanged = False
# hack so al_analysis can edit this to add functions i currently have defined under main
# (e.g. save_method_csvs)
# TODO just for -c/-C right? explain better the need for this
_consider_as_main = []

# TODO unit test?
#
# TODO default verbose=None and try to use default of wrapped fn then
# (or True otherwise?)
# (still need to test behavior when wrapped fn has existing verbose kwarg)
_fn_name2seen_inputs: Dict[str, Path] = dict()
_fn_name2wrapped_fn: Dict[str, Callable] = dict()
# TODO also include savefig plots in these (partially to make replacing custom logic in
# there w/ wrapping it w/ @produces_output easier later)
_all_seen_inputs: Set[Path] = set()
CodeContext = namedtuple('CodeContext', 'filename lineno fn_name')
_saved_path2last_save_code_context: Dict[Path, CodeContext] = dict()
# TODO what is _fn for again? keep?
def produces_output(_fn=None, *, verbose: bool = True):
    """Decorator for fns saving `data` to `path`, called as `fn(data, path, **kwargs)`

    Can be used either as `@produces_output` or `@produces_output(verbose=False)`.

    The wrapped fn:
    - also accepts any `Pathlike` for `path`

    - raises `MultipleSavesPerRunException` if the same (resolved) path is written more
      than once in a run (by any wrapped fn), unless `multiple_saves_per_run_ok=True`

    - if `check_outputs_unchanged` (set by -c/-C CLI flags) and `path` exists,
      checks that the new output would not change it (not writing it, if unchanged).
      See `ignore_output_change_check` for how changes are handled.

    Args:
        _fn: the fn to wrap, when used without parentheses (`@produces_output`)

        verbose: default for `verbose` kwarg of wrapped fn (whether to print path before
            writing)
    """
    # for how to make a decorator with optional  arg:
    # https://realpython.com/primer-on-python-decorators

    # TODO what would be a good name for this?
    def wrapper_helper(fn):
        """Returns `fn` wrapped as described in `produces_output` docstring.
        """
        assert fn.__name__ not in _fn_name2seen_inputs, (
            f'{fn.__name__=} seen set would have been overwritten'
        )
        # TODO some reason to use lists like i was in savefig? was that just for easier
        # use in multiprocessing access (no set equiv of IPC data type?)?
        # that matter anymore?
        _fn_name2seen_inputs[fn.__name__] = set()

        @wraps(fn)
        # TODO delete *args (if assertion it's unused passes for a wihle)
        def wrapped_fn(data, path: Pathlike, *args, verbose: bool = verbose,
            ignore_output_change_check: Union[bool, str] = False,
            multiple_saves_per_run_ok: bool = False, **kwargs):
            """
            Args:
                verbose: if True, prints `path` before writing it

                ignore_output_change_check: only relevant if checking outputs are
                    unchanged (-c/-C). if True, silently overwrites changed outputs. if
                    'warn', warns before overwriting. if False, raises an error (or
                    prompts, if -P was passed).

                multiple_saves_per_run_ok: if False, will raise
                    MultipleSavesPerRunException if same absolute path is written to
                    twice in one run (which is assumed to usually be a programming
                    mistake)

                **kwargs: passed to wrapped fn
            """
            # TODO delete (probably delete *args in sig above if so)
            assert len(args) == 0
            #

            assert ignore_output_change_check in (True, False, 'warn')

            # TODO easy to check type of matching positional arg is Path/Pathlike
            # (if specified)?
            # see: https://stackoverflow.com/questions/71082545 for one way

            path = Path(path)

            # so that no matter how input is specified, check on whether we already
            # wrote to it (via `seen_inputs`) not have any false negatives.
            #
            # from `Path.resolve` docs:
            # "Make the path absolute, resolving any symlinks."
            # "'..' components are also eliminated"
            normalized_path = path.resolve()

            # TODO add option (for use during debugging) that checks outputs
            # have not changed since last run (to the extent the format allows it...)
            # (currently have this via global `check_outputs_unchanged`. do i actually
            # want it to specifically be a kwarg added by the wrapper instead?)

            assert fn.__name__ in _fn_name2seen_inputs
            # TODO probably don't want different fns to be able to save to same path
            # either tho... (not that they currently would). maybe seen_inputs should be
            # one global?
            seen_inputs = _fn_name2seen_inputs[fn.__name__]

            if not multiple_saves_per_run_ok and (
                (normalized_path in seen_inputs or normalized_path in _all_seen_inputs)
                ):
                context = _saved_path2last_save_code_context[normalized_path]
                raise MultipleSavesPerRunException('would have overwritten output '
                    f'{path}\npreviously written elsewhere in this run, at:\n'
                    f'{context.filename}, line {context.lineno} (in {context.fn_name})'
                    '\nadd multiple_saves_per_run_ok=True to call to override, but this'
                    ' is likely a mistake'
                )

            # TODO TODO try to only add these after successfully saving
            seen_inputs.add(normalized_path)
            # TODO why did i need a fn specific cache anyway? paths should be unique
            # across all fns anyway, right? i don't want two fns overwriting each others
            # outputs within a run... simplify by replacing all fn specific sets w/ this
            # one global one?
            _all_seen_inputs.add(normalized_path)

            last_frame = inspect.currentframe().f_back
            # "index" (last argument) is "the index of the current line being executed
            # in the code_context list", but not sure what that means, and don't seem to
            # need it for what i want
            filename, lineno, fn_name, _, _ = inspect.getframeinfo(last_frame)
            _saved_path2last_save_code_context[normalized_path] = CodeContext(
                filename=filename, lineno=lineno, fn_name=fn_name
            )

            write_output = True
            if check_outputs_unchanged and path.exists():
                try:
                    _check_output_would_not_change(path, fn, data, **kwargs)
                    return

                except RuntimeError as err:
                    # TODO also need to apply this logic in savefig? or just replace
                    # savefig w/ a simpler version (wrapped w/ @produces_output)?
                    # no plots really saved directly under main (at least not
                    # unconditionally...)
                    calling_fn = caller_info().name
                    if calling_fn == 'main' or calling_fn in _consider_as_main:
                        if check_outputs_unchanged == 'nonmain':
                            # TODO refactor to share mtime/etc parts of message w/ other
                            # main places similar message (w/ 'would have changed')
                            # defined? not sure it matters enough here...
                            warn(f'{path} would have changed!\nnot failing because -C '
                                'passed (instead of -c), and this output saved directly'
                                ' in main().\nremove -C/-c and re-run to overwrite.'
                            )
                            # TODO option to not return (which would allow the `fn` call
                            # below to overwrite this output)?
                            return

                        elif check_outputs_unchanged == True:
                            assert len(err.args) == 1
                            curr_msg = err.args[0]
                            err = RuntimeError(f'{curr_msg}\n\nor pass -C instead of '
                                '-c, to ignore outputs (like this one) saved directly '
                                'from main()'
                            )

                    # TODO add kwarg to only ignore if diff matches a certain value?
                    _warn_only = False
                    if ignore_output_change_check and check_outputs_unchanged != False:
                        _warn_only = True

                    # 'warn'|False will still enter this conditional
                    if ignore_output_change_check != True:
                        write_output = _output_change_prompt_or_err(err, path,
                            _warn_only=_warn_only
                        )

            if write_output:
                # TODO test! (and test arg kwarg actually useable on wrapped fn, whether
                # or not already wrapped fn has this kwarg. can start by assuming it
                # doesn't have this kwarg tho...)!
                #
                # (have already manually tested cases where wrapped fns do not have
                # existin verbose= kwarg. just need to test case where wrapped fn DOES
                # have existing verbose= kwarg now.)
                if verbose:
                    print(f'writing {path}')

                fn(data, path, **kwargs)

        _fn_name2wrapped_fn[fn.__name__] = wrapped_fn
        return wrapped_fn

    # TODO what is this for again?
    if _fn is None:
        return wrapper_helper
    else:
        return wrapper_helper(_fn)


# if one output would get written two twice in one run of this script.
# for most outputs, we only intend to write them once, and this indicates an error.
class MultipleSavesPerRunException(IOError):
    pass


# set True by al_analysis if -P passed as CLI arg
prompt_if_changed: bool = False

def _output_change_prompt_or_err(err: RuntimeError, path: Path, *,
    _warn_only: bool = False) -> bool:
    """Returns whether `path` should be written to with new version.

    If `prompt_if_changed = True`, prompts user to choose what to do with file
    (whether to overwrite/skip/exit). If `prompt_if_changed = False`, raises
    `err` passed in.

    Args:
        _warn_only: if True, just warns with message of `err` and returns True (no
            prompt or error)
    """
    assert err is not None
    if not prompt_if_changed and not _warn_only:
        raise err

    assert len(err.args) == 1
    msg = err.args[0]
    # TODO still show lineno (of fn calling wrapped_fn) in this case?
    warn(msg)
    if _warn_only:
        return True

    overwrite = None
    # TODO try to have Ctrl-c behave same as 'n'?
    while True:
        # TODO option to accept all future prompts too? (would still want to warn for
        # each)
        # TODO maybe for anything <= current rms (prompting again if new max)?
        # TODO option to start debugger here (to step up and inspect difference)?
        response = input(f'overwrite {path}?\n[y]es / [n]o (quit) / [s]kip (continue) '
            '(press enter after selection)'
        )
        response = response.lower()

        if response == 'y':
            overwrite = True
            break

        elif response == 'n':
            raise err

        # [s]kip
        elif response == 's':
            overwrite = False
            break
        else:
            print('invalid response. expected y/n/s.')

    assert overwrite is not None
    return overwrite


def caller_info() -> traceback.FrameSummary:
    """Returns FrameSummary for the caller of the fn that calls `caller_info`.

    FrameSummary objects have attributes: filename, lineno, name, line (as well as
    locals).
    """
    stack = traceback.extract_stack()
    assert len(stack) >= 4
    # NOTE: .name of each element is the function name of that element
    # of the stack. if at module level (not within a function), .name
    # should be (literal, not w/ actual module name) '<module>'. not necessarily the
    # first element, as e.g. `python -m pytest` will have runpy frames before that.
    assert any(x.name == '<module>' for x in stack)
    assert stack[-1].name == 'caller_info' and not any(
        x.name == 'caller_info' for x in stack[:-1]
    )
    # n_levels_up=1 would just put us in the caller of caller_info, which is not useful.
    # we want the caller of caller_info's caller.
    # TODO expose as kwarg? no use less than 2 tho
    n_levels_up = 2
    frame_summary = stack[-1 - n_levels_up]
    return frame_summary


@produces_output
# input could be at least Series|DataFrame
def to_csv(data: DataFrameOrSeries, path: Path, **kwargs) -> None:
    """Writes `data` to CSV at `path` (which must end in '.csv'), via `data.to_csv`.

    NOTE: `produces_output` wrapper modifies fn to allow `Pathlike` for path arg
    """
    assert path.suffix == '.csv', f"{path.suffix} should be '.csv'"
    data.to_csv(path, **kwargs)


_SERIES_NAME_PLACEHOLDER: str = '__UNNAMED-SERIES'
# TODO some nice way to make a list of all these fns and then redef all of them w/
# wrapper? or too meta-programmy (don't want to eval...)?
def read_parquet(path: Path, *, squeeze: bool = True) -> DataFrameOrSeries:
    """Reads parquet written by `to_parquet`, undoing its conversions.

    Moves any sequence-valued first column (from a tuple-valued last index level, e.g.
    bouton IDs) back to a final index level of tuples, and converts column MultiIndex
    level values back to numeric / datetime, where possible.

    Args:
        squeeze: if True, returns a Series if data has only one column. Series written
            without a name are returned as (unnamed) Series either way.
    """
    # TODO try changing default engine (='fastparquet'? or ='pyarrow'? latter should be
    # default, if both installed [at least in pandas 3.0...])? that fix any of column
    # level type handling i add below?
    data = pd.read_parquet(path)

    # TODO need an outer need len check? prob not practically, but maybe for some edge
    # cases, if trying to support saving empty stuff?
    # TODO refactor to share w/ similar code in to_parquet?
    move_first_column_to_index = False
    i00 = data.iloc[0, 0]
    # TODO also exclude anything that isn't explicitly a sequence of float/int?
    if not isinstance(i00, str):
        try:
            len(i00)
            move_first_column_to_index = True
        except TypeError as err:
            assert str(err).endswith('has no len()')
    #

    # TODO assert no sequence values in index already? (would not be possible here, b/c
    # read_parquet above would fail. may want something like that in to_parquet)

    if move_first_column_to_index:
        assert len(data.columns) > 1, \
            'still need some columns left after moving one to index'

        c0 = data.columns[0]
        # we need to make sequences hashable to move to index. tuple is the way i had
        # done this with bouton_ids, and the most straightforward.
        data[c0] = data[c0].map(tuple)
        data = data.set_index(c0, append=True)

    # TODO do for non-MultIndex indices as well? (only [currently] converting levels on
    # save for MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        index_arrays = []
        for c in data.columns.names:
            xs = data.columns.get_level_values(c)
            # TODO anything other types besides numeric/datetime?
            try:
                # seems to automatically convert to int if it can (haven't tested float,
                # especially not with NaN, but assume that would remain float too)
                xs = pd.to_numeric(xs)

            # ValueError: Unable to parse string <x> at position <n>
            except ValueError:
                try:
                    xs = pd.to_datetime(xs)
                except ValueError:
                    pass

            # each will still have .name defined
            index_arrays.append(xs)

        data.columns = pd.MultiIndex.from_arrays(index_arrays)

    # all Series written with to_parquet above (if they had None for .name), should have
    # this as last column
    if data.columns[-1] == _SERIES_NAME_PLACEHOLDER:
        data = data.squeeze().rename(None)

    # TODO remove kwarg? always squeeze? would then prob want to assert no DataFrame
    # inputs with only one column (which there probably aren't...)
    elif len(data.columns) == 1 and squeeze:
        return data.squeeze()

    return data


@produces_output(verbose=True)
def to_parquet(data: DataFrameOrSeries, path: Path, *, check: bool = True) -> None:
    """Write `data` to parquet at `path`, with default check loaded value matches input.

    Supports some data parquet can't directly store, converting it in ways that
    `read_parquet` undoes: Series (including those without a name), non-str column
    MultiIndex level values (converted to str), and a tuple-valued last index level
    (moved to first column).

    Args:
        check: if True, reads output back (via `read_parquet`) and checks it matches
            `data`
    """
    assert path.suffix == '.parquet', f"{path.suffix} should be '.parquet'"
    # TODO make check=False the default eventually?
    if check:
        orig = data

    if isinstance(data, pd.Series):
        assert not hasattr(data, 'to_parquet')
        # since DataFrame.to_parquet(...) requires str column names, but we often have
        # Series with .name == None.
        # Not planning to support non-str for column names other than just a single
        # missing Series name.
        if data.name is None:
            data = data.to_frame(name=_SERIES_NAME_PLACEHOLDER)
        else:
            data = data.to_frame()

    # TODO remove outer len check?
    if len(data) > 0:
        i00 = data.iloc[0, 0]
        # TODO also exclude anything that isn't explicitly a sequence of float/int?
        if not isinstance(i00, str):
            try:
                len(i00)
                raise ValueError('data already had sequence elements in first column. '
                    'would be inferred as a final tuple-of-int index level with current'
                    ' scheme'
                )
            except TypeError as err:
                assert str(err).endswith('has no len()')

    # pd.MultiIndex objects have levels. non-MultiIndex pd.Index objects do NOT.
    if hasattr(data.columns, 'levels'):
        str_levels = [all(type(x) is str for x in xs) for xs in data.columns.levels]

        # TODO some way to avoid the copy here? (without changing input)
        data = data.copy()

        # TODO assert all non-str are all int/float? (basically, some list of types we
        # should be able to detect and convert back on read)
        # TODO try to support datetime/Timestamp here too? anything else?
        data.columns = data.columns.map(lambda xs:
            tuple(x if was_str else str(x) for x, was_str in zip(xs, str_levels))
        )
        # TODO  assert no already-str levels would be detected convertible to int/float
        # (/whatever other types we are converting in read) here?
    else:
        # NOTE: currently assuming any non-MultIndex input will already just have str
        # columns (could implement without too much effort)
        assert all(type(x) is str for x in data.columns)

    # TODO how would this interact with MultiIndex columns? assert we don't have both?
    # test some cases there?
    #
    # currently only planning on supporting automatic tuple-of-int (index) <->
    # array-of-int (column, as supported by my current pd.read_parquet) conversion for a
    # final index level (e.g. bouton_id, which I'm currently assuming will always be
    # last level, when present).
    #
    # I assume that, for anything more complicated than one level at the end of index,
    # the easiest thing to do will be to add an argument to allow caller to specify
    # which columns to convert (and probably need another one for column order, at least
    # so long as I'm doing automated round trip checks, since how else will we know what
    # order to put columns back into index in?)
    if type(data.index.get_level_values(-1)[0]) is tuple:
        # moving last index level to first column
        data = data.reset_index(level=-1)

    # TODO accept engine argument here? changing engine actually avoid any of my current
    # need for custom conversions in to_parquet/read_parquet?
    data.to_parquet(path)

    if check:
        d2 = read_parquet(path)
        try:
            assert d2.equals(orig)
        # TODO TODO was i doing this was just b/c of need for isclose in index? why can
        # we assert values are equal and not just close then?
        except AssertionError:
            # TODO warn here?

            # TODO also convert indices to frames and check those w/ allclose (like
            # columns)? or is parquet MultiIndex reading not broken for those? i had to
            # add manual type conversions for MultiIndex column level values in my
            # read_parquet fn.
            assert d2.index.equals(orig.index)

            c1 = orig.columns.to_frame(index=False)
            c2 = d2.columns.to_frame(index=False)
            # TODO handle case where date is datetime64[ns] in one, and object in the
            # other. (now that i added datetime column level handling in read_parquet,
            # should be fine. delete)
            # TODO is it just float levels that are allclose that are causing failure
            # for claws_sims_[sums|maxs]? (seems so, yes)
            assert pd_allclose(c2, c1)

            # TODO TODO fix. failing w/ consensus_df saving w/:
            # ./al_analysis.py -d pebbled -n 6f -t 2023-04-22 -e 2024-01-05 -s corr,intensity,ijroi,model-seeds,model-sensitivity -v -i model -M
            # (equal_nan=True here seems to fix it, but pd_allclose(d2, orig,
            # equal_nan=True) does *not* work? how come?)
            try:
                assert np.array_equal(d2.values, orig.values, equal_nan=True)
            except AssertionError:
                breakpoint()


@produces_output(verbose=False)
# input could be at least Series|DataFrame
def to_pickle(data: Any, path: Path) -> None:
    """Writes input to pickle at `path`.

    NOTE: `produces_output` wrapper modifies fn to allow `Pathlike` for path arg
    """
    # just checking we aren't accidentally calling to_pickle with a path we intend for
    # one of the other calls
    assert path.suffix not in ('.csv', '.parquet', '.nc', '.json')

    # (only checking for xarray objects if xarray was already imported, as data can not
    # be a DataArray otherwise)
    xr = sys.modules.get('xarray')
    if xr is not None and isinstance(data, xr.DataArray):
        path = Path(path)
        # read via: pickle.loads(path.read_bytes())
        # (note lack of need to specify protocol)
        # just specifying protocol b/c docs say it is (sometimes?) much faster
        path.write_bytes(pickle.dumps(data, protocol=-1))
        return

    # NOTE: objects with a `to_pickle` method (e.g. DataFrame/Series) used to also be
    # written via that method first, but that output was always overwritten here
    path.write_bytes(pickle.dumps(data))


def read_pickle(path: Pathlike) -> Any:
    path = Path(path)
    return pickle.loads(path.read_bytes())


@produces_output
def to_json(data: Dict[str, Any], path: Path, *, check: bool = True) -> None:
    """Write `data` to JSON at `path`, with default round trip check.
    """
    _to_json(data, path, check=check)


@produces_output(verbose=False)
def np_save(data: np.ndarray, path: Path, **kwargs) -> None:
    # TODO add check=True option to round trip test load the data?
    """
    NOTE: opposite order of args to `np.save`, which has path first and data second.
    necessary to work w/ my `produces_output` wrapper.
    """
    np.save(path, data, **kwargs)


# name of the JSON al_analysis writes alongside its response CSVs (e.g.
# `ij_certain-roi_stats.csv`), with the parameters of the response calculation
response_calc_params_json_name: str = 'response_calc_params.json'


# TODO maybe don't require is_pair? panel?
# TODO rename?
#
# can have more 'odor<x>' row index levels if there are multiple odors presented at once
# (via air mixing). will have a number of levels equal to maximum presented at once.
required_index_levels: List[str] = ['panel', 'is_pair', 'odor1', 'repeat']

# TODO provide fn to invert zero filling i had done for some new outputs (dropping
# glomeruli w/ all 0s or NaN)

def drop_old_odor_index_levels(df: pd.DataFrame, *, warn_: bool = True
    ) -> pd.DataFrame:
    """Drops binary-mixture-experiment index levels, if they are uninformative.

    Drops 'is_pair' level if it is all False, and 'odor2' level if it is all
    `solvent_str`.

    Args:
        warn_: if True, warns if either level is present but can't be dropped

    Returns `df`, which is also modified in place.
    """
    # for dropping metadata intended for binary mixture experiments
    to_drop = []

    if 'is_pair' in df.index.names:
        if set(df.index.get_level_values('is_pair')) == {False}:
            to_drop.append('is_pair')
        else:
            if warn_:
                warn('index had some is_pair=True entries! not dropping is_pair level!')

    if 'odor2' in df.index.names:
        if set(df.index.get_level_values('odor2')) == {solvent_str}:
            to_drop.append('odor2')
        else:
            if warn_:
                warn(f'index had some odor2 != {solvent_str} entries! not dropping '
                    'odor2 level!'
                )

    if len(to_drop) > 0:
        df.index = df.index.droplevel(to_drop)

    return df


# TODO rename to include dff in name? or antennal? it's not just the counterpart to my
# wrapped to_csv fn, which is intended to writing general CSVs, not just this specific
# format of them
# TODO (delete?) tho should i test this fn can read CSVs written by my to_csv fn?
# TODO can it also be a series? should output be DataFrameOrSeries?
def read_csv(csv: Pathlike, *, drop_old_odor_levels: bool = True,
    check_vs_pickle: bool = True, warn_: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
    """Loads CSV of responses (e.g. `ij_certain-roi_stats.csv`), of odor X fly-ROI shape

    Returns DataFrame with row index levels like ['panel', 'is_pair', 'odor1', 'odor2',
    'repeat'] ('odor<N>' levels after 'odor1' only for air-mixed components), and column
    levels `flyroi_cols` ('date' parsed to datetime, 'fly_num' to int). E.g.:

    date                                           2024-09-03
    fly_num                                                 1
    roi                                                     D    DA1    DA2
    panel                 odor1     odor2   repeat
    glomeruli_diagnostics 3mtp @ -5 solvent 0          -0.062 -0.025 -0.189
                                            1           0.065 -0.003 -0.072

    Args:
        drop_old_odor_levels: if True, drops 'is_pair' / 'odor2' levels when they are
            uninformative (via `drop_old_odor_index_levels`)

        check_vs_pickle: if True, and a pickle with the same name (but '.p' suffix)
            exists alongside CSV, checks loaded data matches it

        warn_: passed to `drop_old_odor_index_levels`, if called

        verbose: if True, prints path and modification time of CSV (and result of any
            pickle check)
    """
    # TODO does this work on both ij_certain-roi_stats.csv and ij_roi_stats.csv outputs?

    csv = Path(csv)
    assert csv.exists(), f'CSV {csv} did not exist!'

    if verbose:
        print(f'loading {csv}')
        print(f'modified {format_mtime(getmtime(csv), year=True)}')
        print()

    # this line will start with the level names for the row index, e.g.
    # ['panel', 'is_pair', 'odor1', 'repeat', nan, ...]
    # (for data with a maximum of one odor component mixed-in-air)
    #
    # if there are additional components, there will be additional 'odor<N>' index
    # levels, up to the maximum # of components presented at once (via air mixing. odors
    # mixed in vial are represented by their own unique name, not via separate levels).
    # most I've ever used (via air mixing) is 2.
    #
    # not sure if there might be data after these level names, or if it will always be
    # NaN for everything else in that row (but doesn't matter for this).
    index_row = pd.read_csv(csv, skiprows=3, nrows=1, header=None).squeeze()
    # 'repeat' should always be the last level, so if we find that, we know we only have
    # to read index_col up to there (in next read_csv call)
    eq_repeat = index_row == 'repeat'
    # TODO TODO why failing now?
    assert eq_repeat.sum() == 1, f'{eq_repeat.sum()=}'
    # index starts at 0, so we will need to read 1 more index_col level past this
    repeat_idx = eq_repeat.idxmax()
    n_index_col_levels = repeat_idx + 1

    df = pd.read_csv(csv,
        # can't pass list of names here when specifying column MultiIndex via header.
        # trying raises ValueError to that effect.
        index_col=list(range(n_index_col_levels)),

        # doesn't take list of str, and names= seems to only set column level names,
        # rather than finding them by name.
        header=list(range(len(flyroi_cols)))
    )
    # this is assumed in section above that determines how many index levels there are
    assert df.index.names[-1] == 'repeat'

    assert df.columns.names == flyroi_cols
    assert set(required_index_levels) - set(df.index.names) == set()

    # the only row index levels not in required_index_levels should be extra air-mix
    # components (with names in 'odor<N>' form)
    prefix = 'odor'
    for x in set(df.index.names) - set(required_index_levels):
        assert x.startswith(prefix)
        try:
            component_num = int(x[len(prefix):])
        # only intending to catch int parsing failure like:
        # ValueError: invalid literal for int() with base 10: ...
        except ValueError:
            # TODO include better message about malformed index level name
            raise

        # TODO assert all contiguous?
        assert component_num >= 1

    # TODO refactor?
    assert df.columns.names[0] == fly_cols[0]
    df.columns = df.columns.set_levels(pd.to_datetime(df.columns.levels[0]),
        level=0, verify_integrity=True
    )

    assert df.columns.names[1] == fly_cols[1]
    df.columns = df.columns.set_levels(df.columns.levels[1].astype(int), level=1,
        verify_integrity=True
    )

    if check_vs_pickle:
        # just some checking i was doing against a parallel pickle version i had, mainly
        # to make sure i was loading CSV correctly (with same dtype info)
        pickle_path = csv.with_suffix('.p')
        if pickle_path.exists():
            pdf = pd.read_pickle(pickle_path)
            assert pd_allclose(df, pdf, equal_nan=True)
            if verbose:
                print(f'CSV data matches pickle {pickle_path}\n')
        else:
            if verbose:
                print(f'no pickle at {pickle_path}. could not check against CSV.\n')

    if drop_old_odor_levels:
        df = drop_old_odor_index_levels(df, warn_=warn_)

    return df


# TODO unit test this?
def text_diff(f1: Path, f2: Path) -> str:
    """Returns unified diff (w/o context lines) between text of files `f1` and `f2`.
    """
    lines1 = f1.read_text().strip().splitlines(keepends=True)
    lines2 = f2.read_text().strip().splitlines(keepends=True)

    # TODO add git diff style coloring (would conflict w/ how current output is returned
    # as a str which gets often displayed via a `warn` call that already colors output
    # yellow. could return to calling sys.stderr.writelines in here, but would
    # complicate keeping style consistent w/ warning in general)?
    #
    # for reference, default ubuntu diff output on same two files as below:
    # $ diff .../params.csv /tmp/tmph87sfj7j.csv
    # 10c10
    # < used_model_cache,True
    # ---
    # > used_model_cache,False
    #
    # example output:
    # --- .../params.csv
    # +++ /tmp/tmph87sfj7j.csv
    # @@ -10 +10 @@
    # -used_model_cache,True
    # +used_model_cache,False
    #
    # TODO also populate fromdate [+todate?] fields? (w/ mtime? what format?)
    # (seems they take str, so pre-formatted as i want)
    #
    # NOTE: do need to convert path to str for fromfile/tofile
    diff = difflib.unified_diff(lines1, lines2, fromfile=str(f1), tofile=str(f2), n=0)

    # elements in `diff` seem to already have linesep chars at end, so don't need
    # '\n'.join(diff)
    return ''.join(diff)


_save_fn_name2diff_fn = {
    'to_csv': text_diff,
}
# TODO TODO unit test all combinations of changed/unchanged w/ CSV / pickle / some plot
# formats
# TODO option to use np.isclose or something instead of exact file comparison?
# (mainly thinking for CSVs, where the specific way the computation is done might change
# and just lead to a non-important numerical change. hasn't been an issue so far though)
# TODO change so save_fn is the optional one (rather than data) and automatically use
# .savefig if that positional argument has that attribute (or is a Figure/FacetGrid?
# checking for savefig attr probably better...)
# TODO also support objects w/ .save(path) method? (e.g. statsmodels models)
# TODO option to touch files-that-would-be-unchanged to have mtime as if they were just
# written?
def _check_output_would_not_change(path: Path, save_fn: Callable,
    data: Optional[Any] = None, **kwargs) -> None:
    """Raises RuntimeError if output would change.

    Saves new output to a temporary file (via `save_fn`), and compares that to existing
    `path`: exact file comparison (falling back to comparing loaded data, for pickles),
    or matplotlib image comparison, if `save_fn` is a `savefig`. If output would
    change, temporary file is kept, and error message includes a diff (for CSVs) or
    the temporary file path.

    Args:
        path: must already exist (raises IOError if not)

        data: if passed, will be 1st `save_fn` argument, otherwise file path is 1st
            argument, and it's assumed that the data to save is in `kwargs`

        **kwargs: passed to `save_fn`
    """
    # TODO also raise if not file? don't think anything below is equipped to handle e.g.
    # a directory
    if not path.exists():
        raise IOError(f'{path} did not exist!')

    # TODO derive name in deterministic way from path, so that same input will always
    # overwrite any pre-existing temp outputs? not a huge priority, just some temp files
    # could get left around as-is (which is probably fine. generally get deleted on
    # reboot)
    #
    # TODO test w/ input where output filename doesn't already have suffix?
    # support that?
    # (if input doesn't have '.' in name, suffix is ''. if input name starts with '.',
    # and there isn't another '.', suffix is also '')
    #
    # need to use existing suffix when savefig is matplotlib Figure.savefig, as it will
    # save to `temp_file + '.png'` instead of `temp_file`.
    # path.suffix for e.g. Path('x/y/z.pdf') is '.pdf'.
    temp_file = NamedTemporaryFile(delete=False, suffix=path.suffix)
    # also includes directory
    temp_file_path = Path(temp_file.name)

    # TODO move some/all of use_mpl_comparison/is_pickle def to just before conditional
    # (as part of factoring out a file comparison fn from within this)
    use_mpl_comparison = False

    # for save_fn input like:
    # <bound method Figure.savefig of <Figure size 1920x1440 with 1 Axes>>
    # I wasn't actually seeing __name__ in dir(save_fn) for that, but accessing
    # .__name__ still worked (providing 'savefig'). __func__ / __self__ may be the
    # function and bound instance, but not sure, and this seems like it might work ok.
    #
    # TODO test w/ seaborn input
    if save_fn.__name__ == 'savefig':
        use_mpl_comparison = True

    is_pickle = False
    if path.suffix == '.p':
        # TODO could also check if save_fn is to_pickle (or has 'pickle' in name, or
        # name split on '_'?). i always name pickles w/ .p, but if anyone else uses this
        # fn (for pickles), might matter.
        # TODO more generally, check if there is a read_<x> fn defined in same scope as
        # input to_<x> fn name (-> use read fn to compare if filecmp fails). or accept
        # read fn as optional input?
        assert not use_mpl_comparison
        is_pickle = True

    # TODO set verbose=False if save_fn already had that?
    # assuming we don't need to for now. any other issues w/ calling wrapped
    # fn twice?
    # TODO possible to refactor to not need to check data (maybe using *args, and
    # changing all fns using produces_output to swap order of data and path args?)?
    if data is not None:
        save_fn(data, temp_file_path, **kwargs)
    else:
        # TODO assert use_mpl_comparison here (and only here?)? (should be only case i'm
        # currently not passing in data, and no real plans for that to change)
        save_fn(temp_file_path, **kwargs)

    temp_file.close()
    # TODO assert some bytes have been written to file?
    # (would have caught save_fn appending suffix issue before i was using existing
    # suffix)

    to_delete = []
    mtime_str = format_mtime(path, year=True, seconds=True)
    err_msg = f'{path} ({mtime_str}) would have changed! (run without -c/-C to ignore)'

    # TODO factor out file comparison fn(s) from this?
    # (-> use to check certain key outputs same as what is committed to repo, e.g.
    # CSVs w/ model responses in data/sent_to_anoop vs current ones)
    # TODO + make available as CLI?

    if not use_mpl_comparison:
        # https://stackoverflow.com/questions/1072569
        unchanged = filecmp.cmp(path, temp_file_path, shallow=False)

        # could also *always* do this comparison, instead of filecmp above, but I'm
        # assuming actually loading the pickles is more expensive than the above.
        # if the filecmp approach *usually* works for pickles (as it seems to), then
        # it's probably better to only load+compare the pickles if that check fails.
        if is_pickle and not unchanged:
            # TODO these read_pickle fns always return output consistent w/
            # pd.read_pickle (in only case checked so far, yes)?
            old = read_pickle(path)
            new = read_pickle(temp_file_path)

            # TODO TODO refactor to use my hong2p.util.equals now?
            # TODO what about if new has 'equals' and old doesn't (could still get
            # ValueError about truth value ambiguous, if new is series?)
            if hasattr(old, 'equals'):
                unchanged = old.equals(new)
            else:
                # TODO also catch possible ValueError here
                # (why need cast to bool in some but not all cases? Series vs
                # DataFrames?)
                # (this was b/c old and new were both dicts w/ some values that were
                # pd.Series in old and np.array in new)
                # (also got it when new was Series and old was np.array)
                try:
                    unchanged = old == new
                except ValueError:
                    # TODO delete
                    print()
                    print(f'{type(old)=}')
                    print(f'{type(new)=}')
                    import ipdb; ipdb.set_trace()
                    #

                try:
                    unchanged = bool(unchanged)

                # will trigger if old/new are like numpy arrays, like:
                # ValueError: The truth value of an array with more than one element is
                # ambiguous. Use a.any() or a.all()
                #
                # even though the `old.equals(new)` check above should be used for
                # pandas objects (and probably anything else that would have this type
                # of error...), a similarly worded error would be emitted if trying to
                # coerce pandas elementwise comparisons to a single bool. e.g.
                # ValueError: The truth value of a DataFrame is ambiguous. Use a.empty,
                # a.bool(), a.item(), a.any() or a.all().
                except ValueError as err:
                    # TODO also filter on err msg, only trying to coerce check w/ .all()
                    # if msg matches [some parts of] expected err msg?
                    unchanged = np.all(unchanged)
    else:
        # TODO need to install something else for this, in a new env?
        # (might have manually installed before in current one...)
        # (and not sure i've tested this path in a fresh install, since import is
        # conditional)
        # TODO TODO add unit test that hits this path
        # TODO when factoring out file comparison fn, def use_mpl_comparison from
        # conditional import for this b/c it can hang sometimes (for unclear reasons).
        # see comment by original import up top (was there up to f1b56dc4). not sure if
        # the hanging will ever come up again, or how to repro.
        from matplotlib.testing import compare as mpl_compare

        # whether extension is in `mpl_compare.comparable_formats()`?
        # (currently ['png', 'pdf', 'eps', 'svg'])
        #
        # TODO can i remove this? what happens if compare_images gets an input w/ a
        # non-comparable format?
        assert path.suffix[1:].lower() in mpl_compare.comparable_formats()

        # TODO want to actually use tolerance > 0 ever? (reading mpl's code, if tol is
        # 0, np.array_equal is used, rather than mpl_compare.calculate_rms)
        #
        # 2025-02-19: getting some PDFs i can't visually tell apart, w/ rms ~3.84
        # (mb_modeling/hist_hallem.pdf)
        tolerance = 0
        # TODO keep something like this? worried it'd also miss a point meaningfully
        # moving around / similar... (as opposed to weird text spacing changes that
        # seemed non-deterministic)
        #tolerance = 15

        # TODO fix ImageComparisonFailure (can i repro?):
        # ...
        #   File "./al_analysis.py", line 4095, in plot_corr
        #     savefig(fig, plot_dir, prefix, bbox_inches='tight', debug=verbose, **_save_kws)
        #   File "./al_analysis.py", line 2988, in savefig
        #     _check_output_would_not_change(fig_path, save_fn, **kwargs)
        #   File "./al_analysis.py", line 2503, in _check_output_would_not_change
        #     diff_dict = mpl_compare.compare_images(path, str(temp_file_path), tolerance,
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 466, in compare_images
        #     rms = calculate_rms(expected_image, actual_image)
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 363, in calculate_rms
        #     raise ImageComparisonFailure(
        # matplotlib.testing.exceptions.ImageComparisonFailure: Image sizes do not match expected size: (359, 382, 3) actual size (359, 383, 3)

        # compare_images seems to save a png alongside input image, and i don't think
        # there are options to not do that, so i'm just deleting those files after
        #
        # https://matplotlib.org/devdocs/api/testing_api.html#module-matplotlib.testing.compare
        #
        unchanged = None
        try:
            # from docs: "Return None if the images are equal within the given
            # tolerance."
            diff_dict = mpl_compare.compare_images(path, str(temp_file_path), tolerance,
                # TODO remove this, since i couldn't really use it to find files to
                # delete anyway (since it's None if no diff)?
                in_decorator=True
            )

        except ImageComparisonFailure as err:
            # TODO fix
            warn(err)
            #
            unchanged = False

        # paths in diff_dict (for temp_file_path=/tmp/tmpdcsvhhzb.pdf) should look like:
        # 'actual': '/tmp/tmpdcsvhhzb_pdf.png',
        # 'diff': '/tmp/tmpdcsvhhzb_pdf-failed-diff.png',
        # 'expected': 'pebbled_6f/pdf/ijroi/mb_modeling/hist_hallem_pdf.png',

        # TODO try to move the to_delete handling to after this conditional
        # (so i can factor this whole conditional, w/ some of earlier stuff, into fn for
        # just comparing files, not doing any of the temp file creation / cleanup)

        # bit more flexible than Path.with_suffix (how so? just extra '.' chars?
        # delete?)
        def with_suffix(filepath, suffix):
            return filepath.parent / f'{filepath.stem}{suffix}'

        def temp_with_suffix(suffix):
            return with_suffix(temp_file_path, suffix)

        if plot_fmt != 'png':
            # don't think i need to worry about these two if plot_fmt == 'png'
            to_delete.extend([
                temp_with_suffix(f'_{plot_fmt}.png'),
                with_suffix(path, f'_{plot_fmt}.png')
            ])

        if unchanged is None:
            if diff_dict is None:
                unchanged = True
            else:
                # NOTE: <x>_failed-diff[_<plot-format>].png created in this case, but we
                # probably want to keep it for inspection, so not adding to to_delete
                unchanged = False
                err_msg += ('\n\nmatplotlib.testing.compare.compare_images output:\n'
                    f'{pformat(diff_dict)}\n\ndiff image kept for inspection'
                )
                # TODO assert diff_dict['actual'] is same as temp_file_path?

                # TODO also open up (xdg-open / whatever) temp png matplotlib wrote (the
                # diff image)?
                # (and yes, it does convert everything to PNG for this, named like
                # /tmp/tmpa3hi5nxy_pdf-failed-diff.png (for diff),
                # /tmp/tmpa3hi5nxy_pdf.png)

    # want to leave this file around if changed, so we can compare to existing output
    if unchanged:
        to_delete.append(temp_file_path)

    for temp_path in to_delete:
        assert temp_path.exists(), f'{temp_path=} did not exist!'
        temp_path.unlink()

    if unchanged:
        if verbose:
            print(f'{path} would be unchanged')

        return

    # TODO (delete?) move temp file into place, if DOESN'T match (after warning)
    # (would need new value for -c, to warn instead of err. not sure i want)?
    # (also, would have to handle in callers)

    save_fn_name = save_fn.__name__
    if save_fn_name in _save_fn_name2diff_fn:
        diff_fn = _save_fn_name2diff_fn[save_fn_name]
        diff_str = diff_fn(path, temp_file_path)
        err_msg += f'\n\n{diff_str}'
    else:
        err_msg += ('\n\ncompare with what would have been new output, saved at: '
            f'{temp_file_path}'
        )

    raise RuntimeError(err_msg)


# changed in al_analysis.main (exposed as command line arguments)
# (to a set of strings)
ignore_existing: Union[bool, Set[str]] = False

# TODO distinguish between these two in CLI doc (or delete blanket -i)
# TODO should i even have a blanket '-i' option?
ignore_if_explicitly_requested = ('json',)
also_ignore_if_ignore_existing_true = ('nonroi', 'ijroi', 'suite2p', 'dff2spiking',
    'model'
)
ignore_existing_options = (
    ignore_if_explicitly_requested + also_ignore_if_ignore_existing_true
)

def should_ignore_existing(name: str, explicit_only: bool = False) -> bool:
    """Returns whether existing outputs of step `name` should be ignored (recomputed).

    Based on `ignore_existing` (set by `-i` CLI arg in `al_analysis.py`). Steps
    in `ignore_if_explicitly_requested` are only recomputed if explicitly requested
    (i.e. not when `ignore_existing == True`).

    Args:
        name: step to check whether it should be recomputed.
            see `ignore_existing_options`.

        explicit_only: currently unused (see `ignore_if_explicitly_requested` instead)
    """
    if type(ignore_existing) is bool:

        # Some steps are unlikely enough to need recomputation, we will only do so if we
        # explicitly request that step be recomputed.
        if name in ignore_if_explicitly_requested:
            return False

        return ignore_existing
    else:
        assert name in ignore_existing_options
        # (if it's not a bool, it should be guaranteed to be a set-of-str here)
        return name in ignore_existing # pylint: disable=unsupported-membership-test


# TODO test still works when used from al_analysis (al_analysis fns still using this
# _dirs_to_delete_if_empty, referring to it w/ `hong2p.io.` prefix now)
_dirs_to_delete_if_empty: List[Path] = []
# TODO add test confirming it also makes parents (i think that was the whole point of
# using os.makedirs)
def makedirs(path: Pathlike) -> Path:
    """Make directory if it does not exist, and register for deletion if empty.
    """
    path = Path(path)
    # TODO make sure if we make a directory as well as some of its parent directories,
    # that if (all) the leaf dirs are empty, the whole empty tree gets deleted
    # TODO shortcircuit to returning if we already made it this run, to avoid the checks
    # on subsequent calls? they probably aren't a big deal though...
    os.makedirs(path, exist_ok=True)
    # TODO only do this if we actually made the directory in the above call?
    # not sure we really care for empty dirs in any circumstances tho...
    _dirs_to_delete_if_empty.append(path)
    return path


save_figs = True

# TODO support multiple (don't know if i want this to be default though, cause extra
# time)
# TODO add CLI arg to override this?
plot_fmt: str = os.environ.get('plot_fmt', 'pdf')

exit_after_saving_fig_containing = None
# TODO CLI flag to (or just always?) warn if there are old figs in any/some of the dirs
# we saved figs in (would only want if very verbose...)?
# TODO maybe refactor to automatically prefix path with '{plot_fmt}/' in here?
#
# Especially running process_recording in parallel, the many-figures-open memory
# warning will get tripped at the default setting, hence `close=True`.
#
# TODO why Type[sns.axisgrid.Grid] instead of just sns.axisgrid.Grid?
# (is below comment why? i assume so...)
# sns.FacetGrid and sns.ClusterGrid should both be subclasses of sns.axisgrid.Grid
# (can't seem to import sns.ClusterGrid anyway... maybe it's somewhere else?)
# TODO TODO try to replace logic w/ this decorator
#@produces_output(verbose=False)
_savefig_seen_paths = set()
# TODO possible to use verbose kwarg alongside global verbose?
# (for using this fn in other scripts, would rather have kwarg, but also don't want to
# have to pass verbose=verbose everywhere in here, nor use global as default kwarg,
# which wouldn't be update by CLI arg correctly...)
# TODO try to restore type hint w/ seaborn type. commenting for now to try to be able to
# not import sns, to use some extra C++ debugging tools that it (and scipy) are
# incompatible with
#def savefig(fig_or_seaborngrid: Union[Figure, Type[sns.axisgrid.Grid]],
def savefig(fig_or_seaborngrid,
    fig_dir: Pathlike, desc: str, *, close: bool = True, normalize_fname: bool = True,
    debug: bool = False, multiple_saves_per_run_ok: bool = False,
    ignore_output_change_check: bool = False, fmt: Optional[str] = None, **kwargs
    ) -> Path:
    """Saves figure (or seaborn grid) to `<fig_dir>/<desc>.<fmt>`, returning that path.

    Creates `fig_dir` if needed. Only saves if `save_figs`. Exits after saving,
    if path contains `exit_after_saving_fig_containing` (set by `al_analysis.py
    -F` CLI arg).

    Args:
        desc: name for plot. converted to a valid filename (via
            `hong2p.util.to_filename`), unless `normalize_fname=False`.

        close: if True, closes figure after saving

        debug: if True, prints path to saved figure (otherwise only printed if
            `verbose`)

        multiple_saves_per_run_ok: if False, raises `MultipleSavesPerRunException` if
            the same path is saved to twice in one run

        ignore_output_change_check: if True, skips checking output would not change
            (only checked if `check_outputs_unchanged`, set by -c/-C CLI args)

        fmt: if None, defaults to `plot_fmt` (typically 'pdf')

        **kwargs: passed to `savefig` method of `fig_or_seaborngrid`
    """
    global exit_after_saving_fig_containing

    if fmt is None:
        fmt = plot_fmt

    if fmt == 'pdf':
        kwargs['metadata'] = {
            # TODO delete? test whether this actually improves reproducibility of output
            # files? other blockers on that?
            # TODO anything similar i want for svg, now that i'm also trying that?
            # even needed in current mpl? not referenced in current docs, and not
            # obviously in settings current mpl testing code enforces for tests
            'creationDate': None,

            # TODO some other way to specify (doesn't seem so)? delete
            # matplotlib doesn't seem to set this by default (grepping an example PDF
            # does contain CreationDate, but nothing case-insensitive matching "interp".
            # hopefully this makes the viewers I typically use behave better when
            # rending imshow/matshow plots with many small cells (so I don't need to
            # make huge figures to even have a chance, e.g. 650 inches tall in current
            # plot_aligned_dynamics plot. for that particular plot, this alone didn't
            # seem to help me reduce size. or maybe my reader is ignoring this flag, as
            # apparently they are free to do)
            # https://stackoverflow.com/questions/27512326
            # if trying to specify this way, get
            # Warning: Unknown infodict keyword: 'interpolate'. Must be one of {'Title',
            # 'ModDate', 'Keywords', 'Trapped', 'CreationDate', 'Author', 'Producer',
            # 'Creator', 'Subject'}.
            # and there is no interpolate=<bool> kwarg
            #'interpolate': False
        }
    #

    if normalize_fname:
        prefix = util.to_filename(desc)
    else:
        # util.to_filename in branch above adds a '.' suffix by default
        prefix = f'{desc}.'

    basename = prefix + fmt

    makedirs(fig_dir)
    fig_path = Path(fig_dir) / basename

    # TODO share logic w/ to_csv above (meaning also want to resolve() in
    # produces_output? or no? already doing that?)
    abs_fig_path = fig_path.resolve()

    # duped from produces_output
    last_frame = inspect.currentframe().f_back
    filename, lineno, fn_name, _, _ = inspect.getframeinfo(last_frame)
    _saved_path2last_save_code_context[abs_fig_path] = CodeContext(
        filename=filename, lineno=lineno, fn_name=fn_name
    )
    _all_seen_inputs.add(abs_fig_path)
    #

    # TODO (option to) also save traceback at each, for easier debugging?
    # TODO some way to patch avoid this for testing? (support plot_dir=None?
    # have everything reset _savefig_seen_paths for each offending call? actually
    # just save all figs into distinct plot dirs, even for testing...?)
    if not multiple_saves_per_run_ok and abs_fig_path in _savefig_seen_paths:
        # TODO (delete?) fix:
        # no uncertain ROIs. not generating uncertain_by_max_resp fig
        # done
        # Warning: correlation shapes unequal (in plot_corrs input)! shapes->counts: {(33, 33): 10, (72, 72): 5}
        # some mean correlations will have more N than others!
        # PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)
        # RESTORE CSV SAVING (AFTER DE-DUPING...)
        # writing pebbled_6f/pdf/ijroi/corr_certain_only/control_flies.csv
        # PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)
        # RESTORE CSV SAVING (AFTER DE-DUPING...)
        # writing pebbled_6f/pdf/ijroi/corr_certain_only/kiwi_flies.csv
        # abs_fig_path=PosixPath('/home/tom/src/al_analysis/pebbled_6f/pdf/ijroi/corr_certain_only/kiwi/2024-09-03_1.pdf')
        # desc='2024-09-03_1'
        # > /home/tom/src/al_analysis/al_analysis.py(2581)savefig()
        # -> 2581     _savefig_seen_paths.add(abs_fig_path)
        # ipdb> u
        # > /home/tom/src/al_analysis/al_analysis.py(8313)plot_corrs()
        # -> 8313         fig_path = savefig(fig, panel_dir, fly_plot_prefix)
        context = _saved_path2last_save_code_context[abs_fig_path]
        # TODO maybe i do need more context (previous stack frames).
        # current issue is same line in two calls to the same fn.
        raise MultipleSavesPerRunException('would have overwritten output '
            f'{fig_path}\npreviously written elsewhere in this run, at:\n'
            f'{context.filename}, line {context.lineno} (in {context.fn_name})'
            '\nadd multiple_saves_per_run_ok=True to call to override, but this'
            ' is likely a mistake'
        )
    #
    _savefig_seen_paths.add(abs_fig_path)

    _skip_saving = False
    # TODO delete if i manage to restore use of produces_output wrapper around savefig
    # TODO still warn at least, if ignore_output_change_check=True (just wrap this fn in
    # produces_output finally, to not duplicate effort?)
    if not ignore_output_change_check and check_outputs_unchanged and fig_path.exists():
        save_fn = fig_or_seaborngrid.savefig
        try:
            # TODO wrapper kwarg (that would be added to some/all decorations)
            # for disabling this for some calls? e.g. for formats where they can
            # change for uninteresting reasons, like creation time
            _check_output_would_not_change(fig_path, save_fn, **kwargs)
            _skip_saving = True

        except RuntimeError as err:
            overwrite = _output_change_prompt_or_err(err, fig_path)
            _skip_saving = not overwrite
    #

    # TODO should i be passing metadata={'creationDate': None} in pdf case?
    # would that help make diffing easier / more reliable? still needed?
    # https://matplotlib.org/2.1.1/users/whats_new.html#reproducible-ps-pdf-and-svg-output
    if save_figs and not _skip_saving:
        fig_or_seaborngrid.savefig(fig_path, **kwargs)

    fig = None
    if isinstance(fig_or_seaborngrid, Figure):
        fig = fig_or_seaborngrid

    # TODO any other types of seaborn objects we might want to support? don't think i've
    # encounterd any so far...
    # (only checking for seaborn objects if seaborn was already imported, as the input
    # can not be one otherwise)
    elif ('seaborn' in sys.modules and
        isinstance(fig_or_seaborngrid, sys.modules['seaborn'].axisgrid.Grid)):
        fig = fig_or_seaborngrid.fig

    assert fig is not None, (f'{type(fig)=} may not have been an instance of '
        'Figure or sns.axisgrid.Grid (above) (or somehow otherwise fig was None)'
    )

    # verbose flag set True/False by CLI in al_analysis.main
    if (verbose and not _skip_saving) or debug:
        # TODO may shorten to remove first two components of path by default
        # (<driver>_<indicator>/<fmt> in most/all cases)?
        color = 'light_blue'
        print(util._colored(str(fig_path), color))

    # TODO move this into produces_output, and work on all filetypes saved w/ that
    # wrapper?
    if (exit_after_saving_fig_containing and
        exit_after_saving_fig_containing in str(fig_path)):

        warn('exiting after saving fig matching '
            f"'{exit_after_saving_fig_containing}':\n{fig_path}"
        )
        sys.exit()

    # TODO warn if any figs are wider than would fit in a sheet of paper?
    # at least for a subset of figs marked as for publications?
    # (e.g. diagnostic example plot_rois fig i'm working on now)

    if close:
        if fig is None:
            raise ValueError(
                f'cannot close unknown plot object of type={type(fig_or_seaborngrid)}'
            )

        plt.close(fig)

    return fig_path


# TODO TODO use in fn to check whether outputs werer created since proc start
def curr_proc_start_time() -> float:
    """Returns float start time of current process, comparable to `time.time()`
    """
    # seems like this should be in same format/offset as time.time(). creator of package
    # uses it as time.strftime(..., time.localtime(p.create_time())) in an example:
    # https://stackoverflow.com/questions/2598145
    import psutil

    proc = psutil.Process()
    # TODO module-level cache?
    create_time = proc.create_time()
    assert 0 < create_time < time.time()
    return create_time


def written_since_proc_start(path: Pathlike) -> bool:
    """Returns whether path was written since start of current process.

    Also returns False if path does not exist.
    """
    if not Path(path).exists():
        return False

    # getmtime output should also be comparable to time.time()
    return getmtime(path) > curr_proc_start_time()
