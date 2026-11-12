"""Characterization tests of the file-output helpers in `hong2p.io`.

These helpers (`produces_output` and the functions it wraps, `savefig`, `makedirs`,
`should_ignore_existing`, `written_since_proc_start`, the -c/-C/-P CLI flags) were moved
here from `al_analysis.al_util`, which re-exports them. These tests are a copy of
al_analysis's `test/test_characterize_output_helpers.py` (which still tests them through
`al_util`), with `out` here being this module, and the frozen outputs under
`io_test_data/`. Only `test_read_csv` differs, using a generated CSV instead of
al_analysis's data.
"""

import builtins
import inspect
import os
from pathlib import Path
import sys
import time
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt

# the module holding the output helpers (`al_analysis.al_util` in al_analysis's copy of
# these tests)
from hong2p import io as out


matplotlib.use('agg')

frozen_dir: Path = Path(__file__).resolve().parent / 'io_test_data'


def check_frozen(name: str, text: str) -> None:
    """Asserts `text` matches the frozen copy at `io_test_data/<basename of name>.*`.

    (`name` as for al_analysis's `check_frozen`, e.g. 'output_helpers/to_csv_example',
    where the frozen copies are .txt files, which hong2p's .gitignore ignores)
    """
    matches = list(frozen_dir.glob(f'{Path(name).name}.*'))
    assert len(matches) == 1, f'{name}: {matches=}'
    expected = matches[0].read_text()
    assert text == expected, f'{name}: text differs from frozen'


@pytest.fixture
def example_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([('megamat', 'eb @ -3', 0), ('megamat', 'eb @ -3', 1),
            ('megamat', 'ms @ -3', 0)], names=['panel', 'odor1', 'repeat']
    )
    columns = pd.MultiIndex.from_tuples([('2023-04-22', 1, 'DM1'),
            ('2023-04-22', 1, 'VA2'), ('2023-04-23', 2, 'DM1')],
        names=['date', 'fly_num', 'roi']
    )
    values = np.array([[0.1, 1.0 / 3, -2.5], [np.nan, 1e-8, 12345.678901234],
        [0.0, -0.0, 7.0]
    ])
    return pd.DataFrame(values, index=index, columns=columns)


@pytest.fixture
def check_unchanged(monkeypatch):
    """Sets `out.check_outputs_unchanged = True` (as `-c` does) for the test."""
    monkeypatch.setattr(out, 'check_outputs_unchanged', True)
    monkeypatch.setattr(out, 'prompt_if_changed', False)


def _set_old_mtime(path: Path, seconds_ago: float = 1000) -> float:
    t = time.time() - seconds_ago
    os.utime(path, (t, t))
    return os.path.getmtime(path)


def _changed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.iat[0, 0] = 99.0
    return df


# which saves a run records

def test_saves_recorded(tmp_path, monkeypatch):
    """Tests each wrapped save fn records the resolved path and the calling code.
    """
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3, 4]})
    calls = [
        (out.to_csv, df, 'sub/../x.csv'),
        (out.to_parquet, df, 'x.parquet'),
        (out.to_pickle, df, 'x.p'),
        (out.to_json, {'a': 1}, 'x.json'),
        (out.np_save, np.arange(3), 'x.npy'),
    ]
    (tmp_path / 'sub').mkdir()
    for fn, data, rel in calls:
        lineno = inspect.currentframe().f_lineno + 1
        fn(data, rel, verbose=False)

        resolved = Path(rel).resolve()
        assert resolved.exists()
        assert resolved in out._all_seen_inputs
        assert resolved in out._fn_name2seen_inputs[fn.__name__]

        context = out._saved_path2last_save_code_context[resolved]
        assert context.filename == __file__
        assert context.lineno == lineno
        assert context.fn_name == 'test_saves_recorded'

    # recorded under the normalized path, however it was spelled
    assert (tmp_path / 'x.csv').resolve() in out._all_seen_inputs


def test_saves_recorded_across_fns(tmp_path):
    """Tests the same path saved by two different wrapped fns also raises.
    """
    path = tmp_path / 'x.npy'
    out.np_save(np.arange(3), path)
    with pytest.raises(out.MultipleSavesPerRunException):
        out.to_pickle(np.arange(3), path)

    # spelled differently, but the same file
    other_spelling = tmp_path / 'sub' / '..' / 'x.npy'
    (tmp_path / 'sub').mkdir()
    with pytest.raises(out.MultipleSavesPerRunException):
        out.np_save(np.arange(3), other_spelling)


def test_failed_save_still_recorded(tmp_path):
    """Tests a path is recorded as saved even if the save fails (current behavior).

    (`produces_output` adds the path before calling the wrapped fn, so a retry in the
    same run needs `multiple_saves_per_run_ok=True`.)
    """
    path = tmp_path / 'x.txt'
    df = pd.DataFrame({'a': [1.0]})
    with pytest.raises(AssertionError, match="should be '.csv'"):
        out.to_csv(df, path)

    assert not path.exists()
    assert path.resolve() in out._all_seen_inputs
    with pytest.raises(out.MultipleSavesPerRunException):
        out.to_csv(df, path)


def test_multiple_saves_message(tmp_path):
    """Tests the MultipleSavesPerRunException message names the first save's location.
    """
    path = tmp_path / 'x.csv'
    df = pd.DataFrame({'a': [1.0]})
    lineno = inspect.currentframe().f_lineno + 1
    out.to_csv(df, path, verbose=False)
    with pytest.raises(out.MultipleSavesPerRunException) as excinfo:
        out.to_csv(df, path, verbose=False)

    msg = str(excinfo.value)
    assert str(path) in msg
    assert f'{__file__}, line {lineno} (in test_multiple_saves_message)' in msg
    assert issubclass(out.MultipleSavesPerRunException, IOError)


def test_verbose_prints(tmp_path, capsys):
    """Tests which wrapped fns print 'writing <path>' by default, and the override.
    """
    df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
    printed: Dict[str, bool] = dict()
    for fn, data, suffix in [(out.to_csv, df, '.csv'), (out.to_parquet, df, '.parquet'),
            (out.to_pickle, df, '.p'), (out.to_json, {'a': 1}, '.json'),
            (out.np_save, np.arange(2), '.npy')]:

        path = tmp_path / f'{fn.__name__}{suffix}'
        capsys.readouterr()
        fn(data, path)
        printed[fn.__name__] = f'writing {path}' in capsys.readouterr().out

        quiet = tmp_path / f'{fn.__name__}_quiet{suffix}'
        fn(data, quiet, verbose=False)
        assert 'writing' not in capsys.readouterr().out

        loud = tmp_path / f'{fn.__name__}_loud{suffix}'
        fn(data, loud, verbose=True)
        assert f'writing {loud}' in capsys.readouterr().out

    assert printed == dict(to_csv=True, to_parquet=True, to_pickle=False, to_json=True,
        np_save=False
    )


def test_registered_wrapped_fns():
    """Tests the set of functions `produces_output` wraps (on import of the module).
    """
    names = sorted(out._fn_name2wrapped_fn.keys())
    for name in ('to_csv', 'to_parquet', 'to_pickle', 'to_json', 'np_save'):
        assert name in names
        assert out._fn_name2wrapped_fn[name] is getattr(out, name)

    params = inspect.signature(out.to_csv, follow_wrapped=False).parameters
    for kw in ('verbose', 'ignore_output_change_check', 'multiple_saves_per_run_ok'):
        assert kw in params


def test_produces_output_decorator_forms(tmp_path, capsys):
    """Tests `@produces_output` and `@produces_output(verbose=False)` on new fns.
    """
    written: List[Path] = []

    @out.produces_output
    def _char_test_save_a(data, path):
        written.append(path)
        path.write_text(data)

    @out.produces_output(verbose=False)
    def _char_test_save_b(data, path, *, extra=None):
        written.append(path)
        path.write_text(f'{data}{extra}')

    _char_test_save_a('x', str(tmp_path / 'a.txt'))
    assert 'writing' in capsys.readouterr().out
    _char_test_save_b('y', tmp_path / 'b.txt', extra='z')
    assert 'writing' not in capsys.readouterr().out

    # path passed to the wrapped fn as Path, even if str was passed
    assert all(isinstance(p, Path) for p in written)
    assert (tmp_path / 'b.txt').read_text() == 'yz'

    with pytest.raises(AssertionError):
        # a 2nd fn with the same name can not be registered
        @out.produces_output
        def _char_test_save_a(data, path):
            pass

    with pytest.raises(AssertionError):
        _char_test_save_b('y', tmp_path / 'c.txt', 'positional')

    for name in ('_char_test_save_a', '_char_test_save_b'):
        out._fn_name2seen_inputs.pop(name)
        out._fn_name2wrapped_fn.pop(name)


# -c / -C / -P (checking outputs would not change)

def test_check_unchanged_output_not_rewritten(tmp_path, example_df, check_unchanged):
    """Tests unchanged outputs are left alone (not rewritten) with -c.
    """
    for fn, data, suffix in [(out.to_csv, example_df, '.csv'),
            (out.to_pickle, example_df, '.p'), (out.to_json, {'a': [1, 2.5]}, '.json'),
            (out.np_save, example_df.values, '.npy')]:

        path = tmp_path / f'x{suffix}'
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(out, 'check_outputs_unchanged', False)
            fn(data, path, verbose=False)

        mtime = _set_old_mtime(path)
        fn(data, path, verbose=False, multiple_saves_per_run_ok=True)
        assert os.path.getmtime(path) == mtime, f'{fn.__name__} rewrote {path}'


def test_check_changed_output(tmp_path, example_df, check_unchanged):
    """Tests changed outputs under -c: error (default), 'warn' or silent overwrite.
    """
    path = tmp_path / 'x.csv'
    out.to_csv(example_df, path, verbose=False)
    before = path.read_text()
    changed = _changed(example_df)

    with pytest.raises(RuntimeError) as excinfo:
        out.to_csv(changed, path, multiple_saves_per_run_ok=True)

    msg = str(excinfo.value)
    assert f'{path} (' in msg and 'would have changed! (run without -c/-C to ignore)' in msg
    # CSV changes come with a diff of the changed lines
    assert '\n-megamat,eb @ -3,0,0.1,' in msg
    assert '\n+megamat,eb @ -3,0,99.0,' in msg
    assert path.read_text() == before

    with pytest.warns(UserWarning, match='would have changed'):
        out.to_csv(changed, path, multiple_saves_per_run_ok=True,
            ignore_output_change_check='warn'
        )
    assert path.read_text() != before

    changed2 = _changed(changed)
    changed2.iat[1, 1] = 5.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out.to_csv(changed2, path, multiple_saves_per_run_ok=True,
            ignore_output_change_check=True
        )

    assert pd.read_csv(path, header=[0, 1, 2], index_col=[0, 1, 2]).iat[1, 1] == 5.0

    with pytest.raises(AssertionError):
        out.to_csv(changed2, path, multiple_saves_per_run_ok=True,
            ignore_output_change_check='other'
        )


def test_check_changed_pickle_compares_loaded(tmp_path, check_unchanged):
    """Tests pickles that differ in bytes but load equal count as unchanged.
    """
    path = tmp_path / 'x.p'
    data = {'a': 1, 'b': 2}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(out, 'check_outputs_unchanged', False)
        out.to_pickle(data, path)

    # same dict, other insertion order -> different pickle bytes, equal when loaded
    reordered = {'b': 2, 'a': 1}
    mtime = _set_old_mtime(path)
    out.to_pickle(reordered, path, multiple_saves_per_run_ok=True)
    assert os.path.getmtime(path) == mtime

    with pytest.raises(RuntimeError, match='would have changed'):
        out.to_pickle({'a': 1, 'b': 3}, path, multiple_saves_per_run_ok=True)


def test_check_nonmain(tmp_path, example_df, monkeypatch):
    """Tests -C ('nonmain') skips (w/ warning) changed outputs saved directly in main().
    """
    monkeypatch.setattr(out, 'prompt_if_changed', False)
    path = tmp_path / 'x.csv'
    out.to_csv(example_df, path, verbose=False)
    before = path.read_text()
    changed = _changed(example_df)

    def main():
        out.to_csv(changed, path, multiple_saves_per_run_ok=True, verbose=False)

    def not_main():
        out.to_csv(changed, path, multiple_saves_per_run_ok=True, verbose=False)

    monkeypatch.setattr(out, 'check_outputs_unchanged', 'nonmain')
    with pytest.warns(UserWarning, match='not failing because -C passed'):
        main()
    assert path.read_text() == before

    with pytest.raises(RuntimeError, match='would have changed'):
        not_main()

    # fns listed in _consider_as_main are treated like main
    monkeypatch.setattr(out, '_consider_as_main', ['not_main'])
    with pytest.warns(UserWarning, match='not failing because -C passed'):
        not_main()
    assert path.read_text() == before

    # -c: error from main, with a hint about -C
    monkeypatch.setattr(out, 'check_outputs_unchanged', True)
    with pytest.raises(RuntimeError, match='or pass -C instead of -c'):
        main()


@pytest.mark.parametrize('responses,overwrite', [
    (['y'], True), (['s'], False), (['?', 'Y'], True), (['n'], None),
])
def test_prompt_if_changed(tmp_path, example_df, check_unchanged, monkeypatch, capsys,
    responses, overwrite):
    """Tests -P prompt: [y]es overwrites, [s]kip keeps the old file, [n]o raises.
    """
    monkeypatch.setattr(out, 'prompt_if_changed', True)
    path = tmp_path / 'x.csv'
    out.to_csv(example_df, path, verbose=False)
    before = path.read_text()

    remaining = list(responses)
    prompts = []
    def fake_input(prompt=''):
        prompts.append(prompt)
        return remaining.pop(0)

    monkeypatch.setattr(builtins, 'input', fake_input)
    if overwrite is None:
        with pytest.warns(UserWarning), pytest.raises(RuntimeError):
            out.to_csv(_changed(example_df), path, multiple_saves_per_run_ok=True)
        assert path.read_text() == before
    else:
        with pytest.warns(UserWarning, match='would have changed'):
            out.to_csv(_changed(example_df), path, multiple_saves_per_run_ok=True)
        assert (path.read_text() != before) == overwrite

    assert remaining == []
    assert prompts[0].startswith(f'overwrite {path}?')
    if '?' in responses:
        assert 'invalid response. expected y/n/s.' in capsys.readouterr().out


def _parse(monkeypatch, argv: List[str], **kwargs):
    import argparse
    # restored after the test (the fn sets these module globals)
    monkeypatch.setattr(out, 'check_outputs_unchanged', out.check_outputs_unchanged)
    monkeypatch.setattr(out, 'prompt_if_changed', out.prompt_if_changed)
    monkeypatch.setattr(out, 'check_nonmain_outputs_unchanged',
        getattr(out, 'check_nonmain_outputs_unchanged', None), raising=False
    )
    monkeypatch.setattr(sys, 'argv', ['prog'] + argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--other', action='store_true')
    return out.add_check_outputs_unchanged_CLI_flag_and_parse_args(parser, **kwargs)


@pytest.mark.parametrize('argv,nonmain_option,expected', [
    ([], False, (False, False)),
    (['-c'], False, (True, False)),
    (['--check-outputs-unchanged'], False, (True, False)),
    (['-c', '-P'], False, (True, True)),
    (['-c', '--prompt-if-changed', '--other'], False, (True, True)),
    ([], True, (False, False)),
    (['-c'], True, (True, False)),
    (['-C'], True, ('nonmain', False)),
    (['--check-nonmain-outputs-unchanged', '-P'], True, ('nonmain', True)),
])
def test_check_outputs_unchanged_cli_flag(monkeypatch, argv, nonmain_option, expected):
    """Tests the -c / -C / -P flags set the module's check / prompt settings.
    """
    args = _parse(monkeypatch, argv, check_nonmain_option=nonmain_option)
    assert (out.check_outputs_unchanged, out.prompt_if_changed) == expected
    assert args.check_outputs_unchanged == (expected[0] is True)
    assert args.other == ('--other' in argv)
    if nonmain_option:
        # (only defined once the flags have been added. via vars, so the import surface
        # scan does not list it as a module attribute)
        assert vars(out)['check_nonmain_outputs_unchanged'] == (expected[0] == 'nonmain')
        assert args.check_nonmain_outputs_unchanged == (expected[0] == 'nonmain')
    else:
        assert not hasattr(args, 'check_nonmain_outputs_unchanged')


@pytest.mark.parametrize('argv,nonmain_option,error', [
    (['-P'], False, AssertionError),
    (['-P'], True, AssertionError),
    (['-c', '-C'], True, SystemExit),
    (['-C'], False, SystemExit),
])
def test_check_outputs_unchanged_cli_flag_errors(monkeypatch, capsys, argv,
    nonmain_option, error):

    with pytest.raises(error):
        _parse(monkeypatch, argv, check_nonmain_option=nonmain_option)


def test_check_outputs_unchanged_cli_flag_conflict():
    """Tests adding the flags to a parser that already has -c raises ArgumentError."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c')
    with pytest.raises(argparse.ArgumentError):
        out.add_check_outputs_unchanged_CLI_flag_and_parse_args(parser)


# skipping existing outputs (-i)

def test_should_ignore_existing(monkeypatch):
    """Tests `should_ignore_existing` for -i (all), -i <steps> and no -i.
    """
    options = tuple(out.ignore_existing_options)
    assert options == ('json', 'nonroi', 'ijroi', 'suite2p', 'dff2spiking', 'model')
    assert tuple(out.ignore_if_explicitly_requested) == ('json',)

    monkeypatch.setattr(out, 'ignore_existing', False)
    assert not any(out.should_ignore_existing(x) for x in options)

    monkeypatch.setattr(out, 'ignore_existing', True)
    assert {x: out.should_ignore_existing(x) for x in options} == dict(json=False,
        nonroi=True, ijroi=True, suite2p=True, dff2spiking=True, model=True
    )
    # (bool case does not check name)
    assert out.should_ignore_existing('not-a-step') is True

    monkeypatch.setattr(out, 'ignore_existing', {'json', 'model'})
    assert {x: out.should_ignore_existing(x) for x in options} == dict(json=True,
        nonroi=False, ijroi=False, suite2p=False, dff2spiking=False, model=True
    )
    with pytest.raises(AssertionError):
        out.should_ignore_existing('not-a-step')

    # explicit_only is currently unused
    assert out.should_ignore_existing('model', explicit_only=True) is True


def test_makedirs(tmp_path, monkeypatch):
    """Tests `makedirs` makes parents, returns a Path and registers dirs for cleanup.
    """
    registered: List[Path] = []
    monkeypatch.setattr(out, '_dirs_to_delete_if_empty', registered)

    path = tmp_path / 'a' / 'b' / 'c'
    ret = out.makedirs(str(path))
    assert isinstance(ret, Path) and ret == path and path.is_dir()
    # already existing: no error, registered again
    assert out.makedirs(path) == path
    assert registered == [path, path]


def test_written_since_proc_start(tmp_path):
    path = tmp_path / 'x.txt'
    assert not out.written_since_proc_start(path)

    path.write_text('x')
    assert out.written_since_proc_start(path)
    assert out.written_since_proc_start(str(path))

    t = out.curr_proc_start_time() - 10
    os.utime(path, (t, t))
    assert not out.written_since_proc_start(path)
    assert 0 < out.curr_proc_start_time() < time.time()


# file formats written / read back

def test_written_csv_and_json_frozen(tmp_path, example_df):
    """Tests the bytes `to_csv` / `to_json` write match frozen copies.
    """
    csv_path = tmp_path / 'x.csv'
    out.to_csv(example_df, csv_path)
    check_frozen('output_helpers/to_csv_example', csv_path.read_text())

    json_path = tmp_path / 'x.json'
    data = dict(a=1, b=0.1, c=[1, 2.5, 'x'], d=dict(e=None, f=True), g=1.0 / 3)
    out.to_json(data, json_path)
    check_frozen('output_helpers/to_json_example', json_path.read_text())
    assert out.read_json(json_path) == data


def _parquet_roundtrip(tmp_path: Path, data, name: str):
    path = tmp_path / f'{name}.parquet'
    out.to_parquet(data, path, verbose=False)
    return out.read_parquet(path)


def test_parquet_roundtrip(tmp_path, example_df):
    """Tests `to_parquet` / `read_parquet` round trips, including the conversions.
    """
    # non-str column MultiIndex levels (fly_num ints, dates parsed back to datetime)
    df = example_df.copy()
    df.columns = pd.MultiIndex.from_arrays([
            pd.to_datetime(df.columns.get_level_values('date')),
            df.columns.get_level_values('fly_num'),
            df.columns.get_level_values('roi'),
        ], names=df.columns.names
    )
    back = _parquet_roundtrip(tmp_path, df, 'multiindex')
    assert back.columns.equals(df.columns)
    assert back.index.equals(df.index)
    assert np.array_equal(back.values, df.values, equal_nan=True)
    assert pd.api.types.is_integer_dtype(back.columns.get_level_values('fly_num'))
    assert pd.api.types.is_datetime64_any_dtype(back.columns.get_level_values('date'))

    named = pd.Series([1.0, 2.0], index=pd.Index(['a', 'b'], name='x'), name='y')
    back = _parquet_roundtrip(tmp_path, named, 'named_series')
    assert isinstance(back, pd.Series) and back.equals(named) and back.name == 'y'

    unnamed = named.rename(None)
    back = _parquet_roundtrip(tmp_path, unnamed, 'unnamed_series')
    assert isinstance(back, pd.Series) and back.equals(unnamed) and back.name is None

    one_col = pd.DataFrame({'y': [1.0, 2.0]})
    out.to_parquet(one_col, tmp_path / 'one_col.parquet', check=False)
    back = out.read_parquet(tmp_path / 'one_col.parquet')
    assert isinstance(back, pd.Series) and back.equals(one_col['y'])
    back = out.read_parquet(tmp_path / 'one_col.parquet', squeeze=False)
    assert isinstance(back, pd.DataFrame) and back.equals(one_col)

    # tuple-valued last index level (e.g. bouton IDs) <-> first column
    tuples = pd.DataFrame({'v': [1.0, 2.0], 'w': [3, 4]}, index=pd.MultiIndex.from_tuples(
        [('a', (1, 2)), ('b', (3,))], names=['kc', 'bouton_ids']
    ))
    back = _parquet_roundtrip(tmp_path, tuples, 'tuples')
    assert back.equals(tuples)
    assert list(back.index.get_level_values('bouton_ids')) == [(1, 2), (3,)]

    seq_first_col = pd.DataFrame({'s': [[1, 2], [3]], 'v': [1.0, 2.0]})
    with pytest.raises(ValueError, match='sequence elements in first column'):
        out.to_parquet(seq_first_col, tmp_path / 'bad.parquet')


@pytest.mark.xfail(strict=True, raises=AttributeError, reason='to_parquet round-trip '
    'check (check=True, the default) fails for one-column DataFrames, which '
    'read_parquet squeezes to a Series (ejhonglab/al_analysis issue #7)'
)
def test_parquet_one_column_check(tmp_path):
    out.to_parquet(pd.DataFrame({'y': [1.0, 2.0]}), tmp_path / 'x.parquet')


def test_pickle_roundtrip(tmp_path, example_df):
    path = tmp_path / 'x.p'
    out.to_pickle(example_df, path)
    assert out.read_pickle(path).equals(example_df)
    with pytest.raises(AssertionError):
        out.to_pickle(example_df, tmp_path / 'x.csv')


def test_suffix_checks(tmp_path, example_df):
    for fn, data, bad in [(out.to_csv, example_df, 'x.p'),
            (out.to_parquet, example_df, 'x.pq'), (out.to_json, {}, 'x.txt')]:
        with pytest.raises(AssertionError, match='should be'):
            fn(data, tmp_path / bad)


# savefig

def _fig(value: float = 1.0):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, value])
    return fig


def test_savefig_paths(tmp_path, monkeypatch, capsys):
    """Tests `savefig` output path (name normalization, format, dir creation).
    """
    monkeypatch.setattr(out, 'verbose', False)
    fig = _fig()
    path = out.savefig(fig, tmp_path / 'new' / 'dir', 'eb @ -3 (x, y) a/b', fmt='png',
        close=False
    )
    assert path == tmp_path / 'new' / 'dir' / 'eb_-3_x_y_a_b.png'
    assert path.exists()
    assert plt.fignum_exists(fig.number)
    assert path.resolve() in out._all_seen_inputs
    assert path.resolve() in out._savefig_seen_paths

    path2 = out.savefig(fig, tmp_path, 'a b.c', normalize_fname=False, fmt='png')
    assert path2 == tmp_path / 'a b.c.png'
    assert not plt.fignum_exists(fig.number)

    # default format is out.plot_fmt (env var plot_fmt, 'pdf' by default)
    path3 = out.savefig(_fig(), tmp_path, 'default_fmt')
    assert path3.suffix == f'.{out.plot_fmt}'
    assert capsys.readouterr().out == ''

    monkeypatch.setattr(out, 'verbose', True)
    path4 = out.savefig(_fig(), tmp_path, 'verbose', fmt='png')
    assert str(path4) in capsys.readouterr().out

    with pytest.raises(out.MultipleSavesPerRunException):
        out.savefig(_fig(), tmp_path, 'verbose', fmt='png')


def test_savefig_save_figs_false(tmp_path, monkeypatch):
    monkeypatch.setattr(out, 'save_figs', False)
    path = out.savefig(_fig(), tmp_path, 'not_saved', fmt='png')
    assert not path.exists()
    assert path.resolve() in out._savefig_seen_paths


def test_savefig_exit_after(tmp_path, monkeypatch):
    """Tests `exit_after_saving_fig_containing` (-F) exits after saving a matching fig.
    """
    monkeypatch.setattr(out, 'exit_after_saving_fig_containing', 'stop_here')
    out.savefig(_fig(), tmp_path, 'keep_going', fmt='png')
    with pytest.warns(UserWarning, match='exiting after saving fig'), \
            pytest.raises(SystemExit):
        out.savefig(_fig(), tmp_path, 'stop_here', fmt='png')

    assert (tmp_path / 'stop_here.png').exists()


def test_savefig_check_unchanged(tmp_path, check_unchanged, monkeypatch):
    """Tests -c with PNG figures: identical figure not rewritten, changed one raises.
    """
    monkeypatch.setattr(out, 'plot_fmt', 'png')
    path = out.savefig(_fig(), tmp_path, 'fig')
    assert path.suffix == '.png'
    mtime = _set_old_mtime(path)

    out.savefig(_fig(), tmp_path, 'fig', multiple_saves_per_run_ok=True)
    assert os.path.getmtime(path) == mtime

    with pytest.raises(RuntimeError, match='would have changed'):
        out.savefig(_fig(2.0), tmp_path, 'fig', multiple_saves_per_run_ok=True)

    out.savefig(_fig(2.0), tmp_path, 'fig', multiple_saves_per_run_ok=True,
        ignore_output_change_check=True
    )
    assert os.path.getmtime(path) != mtime


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='with -c, the '
    'comparison cleanup assumes figures are in plot_fmt, so a savefig(..., fmt=<other>) '
    'over an existing figure fails looking for temp files of the wrong format '
    '(ejhonglab/al_analysis issue #8)'
)
def test_savefig_check_unchanged_other_fmt(tmp_path, check_unchanged, monkeypatch):
    monkeypatch.setattr(out, 'plot_fmt', 'pdf')
    out.savefig(_fig(), tmp_path, 'fig', fmt='png')
    out.savefig(_fig(), tmp_path, 'fig', fmt='png', multiple_saves_per_run_ok=True)


def test_read_csv(tmp_path):
    """Tests `read_csv` on a response CSV in the format al_analysis writes.

    (al_analysis's copy of these tests reads one of its committed CSVs instead)
    """
    index = pd.MultiIndex.from_tuples([
            ('glomeruli_diagnostics', False, '3mtp @ -5', 'solvent', 0),
            ('glomeruli_diagnostics', False, '3mtp @ -5', 'solvent', 1),
            ('kiwi', False, 'ea @ -4.2', 'solvent', 0),
        ], names=['panel', 'is_pair', 'odor1', 'odor2', 'repeat']
    )
    columns = pd.MultiIndex.from_arrays([
            pd.to_datetime(['2024-09-03', '2024-09-03', '2024-09-04']), [1, 1, 2],
            ['DA1', 'DM1', 'DA1']
        ], names=['date', 'fly_num', 'roi']
    )
    df = pd.DataFrame([[0.1, -0.2, 0.3], [np.nan, 1.5, -0.0], [2.25, 0.0, 1e-3]],
        index=index, columns=columns
    )
    csv = tmp_path / 'ij_certain-roi_stats.csv'
    df.to_csv(csv)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        back = out.read_csv(csv, verbose=False)

    # is_pair all False and odor2 all solvent_str, so both dropped
    assert list(back.index.names) == ['panel', 'odor1', 'repeat']
    assert back.columns.equals(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(back.columns.get_level_values('date'))
    assert pd.api.types.is_integer_dtype(back.columns.get_level_values('fly_num'))
    assert np.array_equal(back.values, df.values, equal_nan=True)

    back2 = out.read_csv(csv, drop_old_odor_levels=False, verbose=False)
    assert list(back2.index.names) == list(df.index.names)
    assert back2.shape == df.shape

    # checked against a pickle with the same name, if there is one
    df.to_pickle(csv.with_suffix('.p'))
    out.read_csv(csv, verbose=False)
    df.iloc[0, 0] = 99.0
    df.to_pickle(csv.with_suffix('.p'))
    with pytest.raises(AssertionError):
        out.read_csv(csv, verbose=False)

    # odor2 not all solvent_str: warns and keeps it
    tuples = list(df.index)
    tuples[2] = ('kiwi', False, 'ea @ -4.2', 'eb @ -3.5', 0)
    df.index = pd.MultiIndex.from_tuples(tuples, names=df.index.names)
    df.to_csv(csv)
    with pytest.warns(UserWarning, match='not dropping odor2 level'):
        back3 = out.read_csv(csv, check_vs_pickle=False, verbose=False)
    assert list(back3.index.names) == ['panel', 'odor1', 'odor2', 'repeat']
