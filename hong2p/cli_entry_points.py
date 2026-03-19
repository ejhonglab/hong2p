
import argparse
from datetime import datetime
import filecmp
import os
from os.path import isdir, exists, join, getmtime
from pathlib import Path
from shutil import copy2
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import List, Optional
import warnings

import requests
import pandas as pd

from hong2p import util, thor
from hong2p.suite2p import print_suite2p_params
from hong2p.util import format_date
from hong2p.viz import showsync


# NOTE: to add additional endpoints:
# 1) Implement a function in here.
#
# 2) Edit __init__.py to also import that function by name.
#
# 3) Edit ../setup.py to add another entry in the list behind 'console_scripts',
#    in the `entry_points` keyword argument to `setup`.


# TODO TODO add entrypoint for diffing odor set between two flies (reporting any odors
# in one but not the other, any concentrations that changed, any solvents that changed)

def thor2tiff_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('thor_raw_dir',
        help='path containing .raw and metadata created by ThorImage'
    )
    # TODO default to just changing extention of input raw? or something like
    # that (or make output name required...)
    parser.add_argument('-o', '--output-name',
        help='full path of .tif to create. raw.tif in same directory by default'
    )
    parser.add_argument('-w', '--overwrite', action='store_true',
        help='otherwise, will fail if output already exists'
    )
    # TODO also expose flip_lr? name can be handled more consistently in code...
    parser.add_argument('-c', '--check-round-trip', action='store_true',
        help='reads created TIFF and checks it is equal to data from ThorImage raw'
    )
    args = parser.parse_args()
    raw_dir = args.thor_raw_dir
    output_name = args.output_name

    # Options are 'err', 'overwrite', 'ignore', or 'load'
    if_exists = 'overwrite' if args.overwrite else 'err'

    check_round_trip = args.check_round_trip

    util.thor2tiff(raw_dir, output_name=output_name, if_exists=if_exists,
        check_round_trip=check_round_trip
    )


def showsync_cli():
    parser = argparse.ArgumentParser()
    # TODO maybe also accept fly / date / thorsync basename instead of this?
    # or also check for the existence of this downstream of `util.raw_data_root`?
    parser.add_argument('thorsync_dir',
        help='path containing output of a ThorSync recording'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
        help='will print all column names as they are in .h5 file and any renaming'
        'that occurs inside thor.load_thorsync_hdf5'
    )
    # TODO say what is show by default (or excluded...)
    parser.add_argument('-a', '--all', action='store_true',
        help='will display all data in HDF5 (except frame counter)'
    )
    parser.add_argument('-d', '--datasets', action='store',
        help='comma separated list of (normalized) names of traces to plot'
    )
    args = parser.parse_args()
    thorsync_dir = args.thorsync_dir
    verbose = args.verbose
    exclude_datasets = False if args.all else None
    datasets = None if not args.datasets else ['gctr'] + args.datasets.split(',')

    showsync(thorsync_dir, verbose=verbose, exclude_datasets=exclude_datasets,
        datasets=datasets
    )


def suite2p_params_cli():
    """Prints data specific parameters so they can be set in suite2p GUI
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('thorimage_dir', nargs='?', default=os.getcwd(),
        help='path containing .raw and metadata created by ThorImage'
    )
    parser.add_argument('-s', '--shape', action='store_true', help='also print movie '
        'shape (for picking registration block size, batch size, etc)'
    )
    args = parser.parse_args()
    print_suite2p_params(args.thorimage_dir, print_movie_shape=args.shape)


def print_dir_fn_cli_wrapper(fn):

    def cli_wrapper():
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', action='store_true')
        args = parser.parse_args()
        verbose = args.verbose

        fn(verbose=verbose)

    return cli_wrapper


# TODO factor my ~/.bash_aliases commands using these (e.g. 2p, 2pr, 2pa, print_2pa)
# into a script in hong2p + call that script to install them in my ~/.bash_aliases
# (+ provide script / instructions for other people to install them)
@print_dir_fn_cli_wrapper
def print_data_root(verbose=False):
    print(util.data_root(verbose=verbose))


@print_dir_fn_cli_wrapper
def print_raw_data_root(verbose=False):
    print(util.raw_data_root(verbose=verbose))


def print_analysis_intermediates_root():
    # This one doesn't take have a `verbose` kwarg like the others
    print(util.analysis_intermediates_root())


def print_paired_thor_subdirs():
    # TODO clarify in doc what "experiment time" is. is it when thorimage started?
    # ended?
    """Prints pairs of (ThorImage, ThorSync) dirs that are direct descendents of input.

    Printed in order of ThorImage experiment times.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default=os.getcwd(),
        help='path containing .raw and metadata created by ThorImage'
    )
    args = parser.parse_args()
    parent = args.dir

    # TODO TODO expose some kwargs to CLI? may need to if we have e.g. 'anat' w/o
    # corresponding ThorSync dir
    paired_dirs = thor.pair_thor_subdirs(parent)

    # TODO opt to include get_thorimage_time in nice format?

    df = pd.DataFrame.from_records(columns=['ThorImage', 'ThorSync'], data=[
        (i.name, s.name) for (i, s) in sorted(paired_dirs, key=lambda p:
        thor.get_thorimage_time(p[0]))
    ])
    print(df.to_string(index=False))


def print_thorimage_subdir_notes():
    """Prints the note field of ThorImage subdirectories.

    Printed in order of ThorImage experiment times.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default=os.getcwd(),
        help='path containing .raw and metadata created by ThorImage'
    )
    args = parser.parse_args()
    parent = args.dir

    thorimage_subdirs = thor.thorimage_subdirs(parent)
    for thorimage_dir in sorted(thorimage_subdirs, key=lambda d:
        thor.get_thorimage_time(d)):

        print(util.shorten_path(thorimage_dir))
        print(thor.get_thorimage_notes(thorimage_dir))
        print('')


def save_requirements() -> None:
    # TODO provide example of url / editable / file:// processing
    # TODO TODO output in conda format (.yaml?), or something else that actually allows
    # installing python at specified version (and ideally pip + setuptools too; the
    # "build" deps)
    """Saves pip requirements to a file, backup up old files at same path.

    Processes git repo auto to https, and including setuptools / pip / Python versions
    in comments at top.

    Args:
        path: if not passed, will save to `requirements.txt` in current directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', nargs='?', default=None,
        help='path output requirements.txt style dependencies will be written to. '
        'if this already exists, it will be backed up to YYYY-MM-DD_<curr-name>. '
        'if not passed, will be written to exact-requirements.txt in current directory.'
    )
    parser.add_argument('-n', '--no-switch-auth', action='store_true', help='if NOT '
        'passed, all auth (e.g. ssh://) will be changed to https://, at least for '
        'public repositories. repositories are assumed private if we get a 404 at their'
        ' URL.'
    )
    parser.add_argument('-a', '--assume-all-public', action='store_true', help='skips '
        'requesting repo URLs, and instead assumes all repos are public (for purpose of'
        ' deciding whether to switch the auth to HTTPS)'
    )
    parser.add_argument('-s', '--no-strip-editable', action='store_true',
        help="if NOT passed, strips '-e ' prefix from lines indicating installed "
        'version is editable'
    )
    parser.add_argument('-d', '--dry-run', action='store_true',
        help='if passed, no files will be written, but -v/--verbose is implied'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    path = args.output_path
    switch_auth_to_https = not args.no_switch_auth
    assume_all_public = args.assume_all_public
    strip_editable = not args.no_strip_editable
    dry_run = args.dry_run
    verbose = args.verbose

    if dry_run:
        verbose = True

    if path is None:
        name = 'exact-requirements.txt'
        parent = Path('.')
        path = parent / name
    else:
        path = Path(path)
        # parent is '.' when path is just a filename
        parent = path.parent
        assert parent.is_dir(), ('path should be a file, and parent directory '
            'should already exist'
        )
        name = path.name

    if not path.is_file():
        # could in theory be a broken symlink/directory, or maybe something else
        assert not path.exists(), (f'{path=} already existed, but was not a file. '
            'delete it, or pass different path.'
        )

    def get_cmd_stdout(cmd: str) -> str:
        cmd_list = cmd.split(' ')
        out = subprocess.run(cmd_list, capture_output=True, text=True)
        if out.returncode != 0:
            # TODO test
            msg = ' '.join(cmd) + ' failed with returncode={out.returncode} and:\n'
            # TODO or would this fail if it's empty?
            stderr = out.stderr
            assert stderr is not None
            msg += stderr
            raise RuntimeError(msg)

        stdout = out.stdout
        assert stdout is not None
        return stdout

    def splitlines(text: str) -> List[str]:
        lines = text.strip().splitlines()
        # TODO or process output lines to strip elements, if this ever failss
        assert all(x.strip() == x for x in lines)
        # TODO filter them, if this fails
        assert all(len(x) > 0 for x in lines)
        return lines

    # TODO if writing to conda yaml (from named conda env), some way to get name of
    # current env?

    freeze_lines = splitlines(get_cmd_stdout('pip freeze'))
    freeze_line_set = set(freeze_lines)
    assert len(freeze_line_set) == len(freeze_lines)

    # these also include "setuptools, pip, wheel, distribute" (from `pip freeze -h`).
    # my current output doesn't seem to include `distribute` though, but i may not just
    # have it installed, or it may be a newer/older thing.
    # can check what should be unique to these in CLI easily via:
    # comm -13 <(pip freeze | sort) <(pip freeze --all | sort)
    freeze_all_lines = splitlines(get_cmd_stdout('pip freeze --all'))
    assert len(set(freeze_all_lines)) == len(freeze_all_lines)

    build_deps = [x for x in freeze_all_lines if x not in freeze_line_set]

    editable_prefix = '-e '
    file_substr = ' @ file://'
    # different auth strings can follow e.g. 'https://<url>', 'ssh://<url>' ...
    # for cases like:
    # `ijroi @ git+https://github.com/tom-f-oconnell/ijroi@65f249bac...`
    noneditable_git_substr = ' @ git+'
    assert not any(
        x.startswith(editable_prefix) or file_substr in x or noneditable_git_substr in x
        for x in build_deps
    )
    # TODO assert only ones mentioned in pip freeze help above?

    # not going to worry about this possibly being part of repo name.
    # some URLs will have this suffix at end of repo name.
    git_suffix = '.git'

    # TODO need to / want to deal w/ git:// auth? still supported? do i still use
    # anywhere?
    # both of these typically follow ' git+' and typically precede 'github.com/<repo>'
    https_auth_str = 'https://'
    # TODO only switch auth to https after testing it's public (by trying to read w/
    # curl or something?)
    ssh_auth_str = 'ssh://git@'

    # '#egg=<repo>' after hash in editable lines I'm currently seeing
    egg_str = '#egg='

    # TODO could assert `pip freeze --exclude-editable` excludes just these?
    editable = []
    noneditable_git = []
    file_lines = []
    regular = []
    for x in freeze_lines:

        # for lines like:
        # `-e git+ssh://git@github.com/ejhonglab/hong2p@7b547afe7...`
        if x.startswith(editable_prefix):
            assert not file_substr in x

            assert noneditable_git_substr not in x, \
                f'{noneditable_git_substr=} was in {repr(x)}'

            editable_git_prefix = f'{editable_prefix}git+'
            # i was imagining changing auth strs for all of the editable ones below
            assert x.startswith(editable_git_prefix), (f'{repr(x)} did not start with '
                '{repr(editable_git_prefix)}'
            )

            assert ssh_auth_str in x, (f'{repr(x)} did not have expected SSH auth str '
                f'{repr(ssh_auth_str)}'
            )

            # TODO why do these ones tend to have '#egg=<repo>' suffix, but non-editable
            # 'ijroi @ git+https://github.com/tom-f-oconnell/ijroi@65f249b...' doesn't?
            # matter if they all do / don't? process?
            parts = x.split('/')

            site_parts = parts[-3].split('@')
            assert len(site_parts) == 2, f'{site_parts=}'
            assert site_parts[0] == 'git', f'{site_parts=}'
            assert len(site_parts[1]) > 0, f'{site_parts=}'
            # should be 'github.com' typically
            site = site_parts[1]

            # e.g. 'ejhonglab'
            account = parts[-2]

            repo = parts[-1]
            # TODO assert '@' in repo? (always have hash following it, in lines i've
            # seen)
            egg_suffix = ''
            commit_suffix = ''
            if '@' in repo:
                repo, commit = repo.split('@')

                # duplicated up here, b/c otherwise repo == repo2 check could fail
                if repo.endswith(git_suffix):
                    repo = repo[:-len(git_suffix)]

                if egg_str in commit:
                    commit, repo2 = commit.split(egg_str)
                    assert repo == repo2, (f'{repo=} (from url) != {repo2=} '
                        '(from egg=<repo>)'
                    )
                    egg_suffix = f'{egg_str}{repo}'
                    # TODO want to strip egg_suffix? option to?

                commit_suffix = f'@{commit}'
            else:
                assert egg_str not in x
                if repo.endswith(git_suffix):
                    repo = repo[:-len(git_suffix)]

            if switch_auth_to_https:
                url = f'{https_auth_str}{site}/{account}/{repo}'
                new = f'{editable_git_prefix}{url}{commit_suffix}{egg_suffix}'

                private_repo = False
                if not assume_all_public:
                    # if URL was not even a valid website, could raise something like:
                    # ```
                    # requests.exceptions.ConnectionError:
                    # HTTPSConnectionPool(host='asdfasdfwerwer.com', port=443): Max
                    # retries exceeded with url: / (Caused by
                    # NameResolutionError("<urllib3.connection.HTTPSConnection object at
                    # 0x7fd47577b970>: Failed to resolve ...
                    # ```
                    r = requests.get(url)
                    # or could be a typo or something, but given it's installed,
                    # probably not. maybe something where package name doesn't match
                    # repo name?
                    private_repo = r.status_code == 404

                if not private_repo:
                    x = new
                else:
                    print('not replacing SSH auth with HTTPS for:\n{x}\n...because '
                        f'repo seems to be private. getting 404 at {url}',
                        file=sys.stderr
                    )

            if strip_editable:
                x = x[len(editable_prefix):]

            editable.append(x)

        elif noneditable_git_substr in x:
            assert f'{noneditable_git_substr}{https_auth_str}' in x, \
                f'auth was not already https: {x}'

            noneditable_git.append(x)

        elif file_substr in x:
            file_lines.append(x)

        else:
            regular.append(x)

    # TODO also include [some part of?] path to venv? / conda env, if set?

    # e.g. '3.8.12'
    python_version = '.'.join(map(str, sys.version_info[:3]))

    lines = [f'# requires-python = "{python_version}"']

    # TODO add CLI flag to exclude build deps? or just comment those lines? (alone w/
    # python version?) not sure it would be helpful in dynamic requirements specified in
    # setuptools section of pyproject.toml (not sure build deps used there. prob need to
    # be prespecified?)
    # TODO TODO add flag to control whether editable things also have '-e ' prefix in
    # output (prob need to default to excluding that)
    lines.append('# build-system.requires:')
    # TODO format in pyproject.toml way? like:
    # [build-system]
    # requires = ["setuptools >= 56.0"]
    lines.extend([f'# {x}' for x in build_deps])

    # TODO add flag to not comment the file_lines entries? not sure when i could install
    # them

    lines.extend(regular)

    # NOTE: some non-editable lines can still be like:
    # ijroi @ git+https://github.com/tom-f-oconnell/ijroi@65f249ba...
    # but probably ok to leave those handled as they are?
    lines.extend(noneditable_git)

    if len(file_lines) > 0:
        msg = ('will not be able to automatically install the following manually'
            ' installed packages:\n'
        )
        msg += '\n'.join(file_lines)
        msg += '\n'
        warnings.warn(msg)

    # TODO possible to get URLs for file:// ones? care to ? need to look at remote in
    # each git repo?
    #
    # commenting because we will not be able to install these on a new system, without
    # manually doing so (e.g. olfsysm)
    lines.extend([f'# {x}' for x in file_lines])

    # TODO editable to local paths? (w/ one CLI flag?)
    # TODO automatically process paths to source dirs for editable installed packages
    # too, to get relative paths to include in dev install file? available in last
    # column of `pip show` output. for what? maybe for getting remote URL, if no other
    # way?
    lines.extend(editable)

    txt = '\n'.join(lines)
    if verbose:
        print(txt)

    def files_match(f1: Path, f2: Path) -> bool:
        t1 = f1.read_text()
        t2 = f2.read_text()
        eq = t1 == t2

        # TODO delete eventually
        # does shallow=False ignore mtime / etc then?
        unchanged = filecmp.cmp(f1, f2, shallow=False)
        assert unchanged == eq, f'{eq=} != {unchanged=}'
        #

        return eq

    backup_msg = None
    if path.is_file():
        temp = NamedTemporaryFile(suffix=f'_{name}')
        temp_path = Path(temp.name, delete=False)
        temp_path.write_text(txt)
        unchanged = files_match(path, temp_path)
        temp.close()
        if unchanged:
            print(f'{path} contents match current output. not making a backup or '
                'writing anything', file=sys.stderr
            )
            return

        # i guess this mtime isn't meaningful when checking out from git. oh well.
        mtime = datetime.fromtimestamp(getmtime(path))
        backup = parent / f'{format_date(mtime)}_{name}'
        if backup.exists():
            unchanged = files_match(backup, path)
            if not unchanged:
                # TODO print diff here?
                raise IOError(f'can not backup current {name}, because {backup} '
                    f'already exists (and contents differ from {path}!)! delete/'
                    'rename it.'
                )
            else:
                print(f'{backup.name} contents match current {path} contents. not '
                    'making an additional backup.', file=sys.stderr
                )
        else:
            if not dry_run:
                # writing to stderr so rest of verbose stdout output could be piped to a
                # file
                print(f'copying existing {name} to backup {backup.name}',
                    file=sys.stderr
                )
                copy2(path, backup)
                assert backup.is_file()
            else:
                print(f'would copy existing {name} to backup {backup.name}',
                    file=sys.stderr
                )

    if not dry_run:
        # writing to stderr so rest of verbose stdout output could be piped to a file
        print(f'writing {path}', file=sys.stderr)
        path.write_text(txt)
    else:
        print(f'would write {path}', file=sys.stderr)

    # TODO add option to save / output in pyproject.toml deps format?
    # (or can pyproject.toml reference requirements.txt like files? maybe, actually?)


