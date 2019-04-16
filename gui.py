#!/usr/bin/env python3

"""
GUI to do make ROIs for trace extraction and validate those ROIs.
"""

import sys
import os
from os.path import split, join, exists, sep, getmtime
import xml.etree.ElementTree as etree
import warnings
from collections import defaultdict
import traceback
from functools import partial
import glob
import hashlib
import time
from datetime import datetime
import socket
import getpass
import pickle
from copy import deepcopy
import pprint
#
import traceback
#

# TODO first three are in QtCore for sure, but the rest? .Qt? .QtGUI?
# differences?
from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot, QThreadPool, QRunnable,
    Qt)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QListWidget,
    QGroupBox, QPushButton, QLineEdit, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QLabel, QListWidgetItem)

import pyqtgraph as pg
import tifffile
import numpy as np
import pandas as pd
# TODO factor all of these out as much as possible
from scipy.sparse import coo_matrix
import cv2
from matplotlib.backends.backend_qt5agg import (FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matlab.engine

# TODO TODO allow things to fail gracefully if we don't have cnmf.
# either don't display tabs that involve it or display a message indicating it
# was not found.
import caiman
from caiman.source_extraction.cnmf import params, cnmf
import caiman.utils.visualization

import util as u


# TODO also move these two variable defs? (to globals below?) for sake of
# consistency?
recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
# TODO use env var like kc_analysis currently does for prefix after refactoring
raw_data_root = '/mnt/nas/mb_team/raw_data'
# TODO support a local and a remote one ([optional] local copy for faster repeat
# analysis)?
analysis_output_root = '/mnt/nas/mb_team/analysis_output'

use_cached_gsheet = True
show_inferred_paths = True
overwrite_older_analysis = True

df = u.mb_team_gsheet(use_cache=use_cached_gsheet)


rel_to_cnmf_mat = 'cnmf'

stimfile_root = '/mnt/nas/mb_team/stimulus_data_files' 

natural_odors_concentrations = pd.read_csv('natural_odor_panel_vial_concs.csv')
natural_odors_concentrations.set_index('name', inplace=True)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() 

def matlabels(df, rowlabel_fn):
    return df.index.to_frame().apply(rowlabel_fn, axis=1)

def odors_label(row):
    if row['name1'] == 'paraffin':
        odors = row['name2']
    elif row['name2'] == 'paraffin':
        odors = row['name1']
    else:
        odors = '{} + {}'.format(row['name1'], row['name2'])
    return odors

def odors_and_repeat_label(row):
    odors = odors_label(row)
    return '{}\n{}'.format(odors, row['repeat_num'])

def thor_xml_root(filename):
    """Returns etree root of ThorImage XML settings from TIFF filename.
    """
    if filename.startswith(analysis_output_root):
        filename = filename.replace(analysis_output_root, raw_data_root)

    parts = filename.split(sep)
    thorimage_id = '_'.join(parts[-1].split('_')[:-1])

    xml_fname = sep.join(parts[:-2] + [thorimage_id, 'Experiment.xml'])
    return etree.parse(xml_fname).getroot()

def cnmf_metadata_from_thor(filename):
    xml_root = thor_xml_root(filename)
    lsm = xml_root.find('LSM').attrib
    fps = float(lsm['frameRate']) / float(lsm['averageNum'])
    # "spatial resolution of FOV in pixels per um" "(float, float)"
    # TODO do they really mean pixel/um, not um/pixel?
    pixels_per_um = 1 / float(lsm['pixelSizeUM'])
    dxy = (pixels_per_um, pixels_per_um)
    # TODO maybe load dims anyway?
    return {'fr': fps, 'dxy': dxy}

def fps_from_thor(df, nas_prefix='/mnt/nas'):
    # TODO assert unique first?
    thorimage_dir = df['thorimage_path'].iat[0]

    if nas_prefix is not None:
        thorimage_dir = join(nas_prefix, *thorimage_dir.split('/')[3:])

    thorimage_xml_path = join(thorimage_dir, 'Experiment.xml')
    xml_root = etree.parse(thorimage_xml_path).getroot()
    lsm = xml_root.find('LSM').attrib
    fps = float(lsm['frameRate']) / float(lsm['averageNum'])
    return fps

def crop_to_nonzero(matrix, margin=0):
    coords = np.argwhere(matrix > 0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    x_min = x_min - margin
    x_max = x_max + margin
    y_min = y_min - margin
    y_max = y_max + margin

    cropped = matrix[x_min:x_max+1, y_min:y_max+1]
    return cropped, ((x_min, x_max), (y_min, y_max))


def closed_mpl_contours(footprint, ax, err_on_multiple_comps=True, **kwargs):
    """
    """
    dims = footprint.shape
    padded_footprint = np.zeros(tuple(d + 2 for d in dims))
    padded_footprint[tuple(slice(1,-1) for _ in dims)] = footprint
    
    mpl_contour = ax.contour(padded_footprint > 0, [0.5], **kwargs)
    # TODO which of these is actually > 1 in multiple comps case?
    # handle that one approp w/ err_on_multiple_comps!
    assert len(mpl_contour.collections) == 1
    paths = mpl_contour.collections[0].get_paths()
    assert len(paths) == 1
    contour = paths[0].vertices

    # Correct index change caused by padding.
    return contour - 1


# TODO worth allowing selection of a folder?
# TODO worth saving labels to some kind of file (CSV?) in case database not
# reachable? fn to load these to database?
# TODO alternative to database for input (for metadata mostly)?

# TODO move natural_odors/kc_analysis.py/plot_traces here?

# TODO TODO support loading single comparisons from tiffs in imaging_util

# TODO TODO TODO factor all code duplicated between here and kc_analysis into a
# util module in this package
def merge_odors(df, odors):
    print('merging with odors table...', end='')
    # TODO way to do w/o resetting index? merge failing to find odor1 or just
    # drop?
    df.reset_index(inplace=True, drop=True)
    #

    df = pd.merge(df, odors, left_on='odor1', right_on='odor',
                  suffixes=(False,False))

    df.drop(columns=['odor','odor1'], inplace=True)
    df.rename(columns={'name': 'name1',
        'log10_conc_vv': 'log10_conc_vv1'}, inplace=True)

    df = pd.merge(df, odors, left_on='odor2', right_on='odor',
                  suffixes=(False,False))

    df.drop(columns=['odor','odor2'], inplace=True)
    df.rename(columns={'name': 'name2',
        'log10_conc_vv': 'log10_conc_vv2'}, inplace=True)

    print(' done')
    return df


def merge_recordings(df, recordings):
    print('merging with recordings table...', end='')

    df.reset_index(inplace=True, drop=True)

    df = pd.merge(df, recordings,
                  left_on='recording_from', right_on='started_at')

    df.drop(columns=['started_at'], inplace=True)

    df['thorimage_id'] = df.thorimage_path.apply(lambda x: split(x)[-1])

    print(' done')
    return df


def motion_corrected_tiff_filename(date, fly_num, thorimage_id):
    date_dir = date.strftime('%Y-%m-%d')
    fly_num = str(fly_num)

    tif_dir = join(analysis_output_root, date_dir, fly_num, 'tif_stacks')

    nr_tif = join(tif_dir, '{}_nr.tif'.format(thorimage_id))
    rig_tif = join(tif_dir, '{}_rig.tif'.format(thorimage_id))
    tif = None
    if exists(nr_tif):
        tif = nr_tif
    elif exists(rig_tif):
        tif = rig_tif

    if tif is None:
        raise IOError('No motion corrected TIFs found in {}'.format(tif_dir))

    return tif


def list_motion_corrected_tifs(include_rigid=False):
    """
    """
    motion_corrected_tifs = []
    for date_dir in glob.glob(join(analysis_output_root, '**')):
        for fly_dir in glob.glob(join(date_dir, '**')):
            try:
                fly_num = int(split(fly_dir)[-1])

                tif_dir = join(fly_dir, 'tif_stacks')
                if exists(tif_dir):
                    tif_glob = '*.tif' if include_rigid else '*_nr.tif'
                    motion_corrected_tifs += glob.glob(join(tif_dir, tif_glob))

            except ValueError:
                continue

    return motion_corrected_tifs
        

# TODO why exactly do we need this wrapper again?
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
        Supplied args and kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # TODO TODO why is this passed as a kwarg? the fn actually used below
        # has progress_callback as the second positional arg...?
        # Add the callback to our kwargs
        #self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()  # Done


class MotionCorrection(QWidget):
    def __init__(self, *args):
        super().__init__(*args)
        if len(args) == 1:
            self.main_window = args[0]

        self.setWindowTitle('Motion Correction')


# TODO either make a superclass for everything that should share a data browser,
# or just have one data browser that interacts with other widgets (tabwidget may
# have to not contain data browser in that case?)
# one argument for trying to have each tab contain a data browser is that then
# each widget (which is currently a tab) definition could more readily by run on
# it's own, without having to combine with a data browser
class Segmentation(QWidget):
    def __init__(self, *args):
        super().__init__(*args)
        if len(args) == 1:
            self.main_window = args[0]

        self.setWindowTitle('Segmentation / Trace Extraction')

        # TODO move initialization of this thing to another class
        # (DataSelector or something) that is shared?
        # TODO TODO TODO use glob or something to find all motion corrected tifs
        # in NAS dir (just show nr?). still display same shorthand for each exp?
        # still use pandas?
        '''
        self.presentations = presentations.set_index(comp_cols)
        # Needs to be indexed same as items, for open_recording to work.
        self.recordings = presentations[recording_cols].drop_duplicates(
            ).sort_values(recording_cols)

        items = ['/'.join(r.split()[1:])
            for r in str(self.recordings).split('\n')[1:]]
        '''
        self.motion_corrected_tifs = list_motion_corrected_tifs()
        for d in self.motion_corrected_tifs:
            print(d)

        entered = pd.read_sql_query('SELECT DISTINCT prep_date, ' +
            'fly_num, recording_from, analysis FROM presentations', u.conn)
        # TODO TODO check that the right number of rows are in there, otherwise
        # drop and re-insert (optionally? since it might take a bit of time to
        # load CNMF output to check / for database to check)

        # TODO more elegant way to check for row w/ certain values?
        '''
        curr_entered = (
            (entered.prep_date == date) &
            (entered.fly_num == fly_num) &
            (entered.recording_from == started_at)
        )
        curr_entered = curr_entered.any()
        '''

        # TODO maybe make slider between data viewer and everything else?
        # sliders for everything in this layout?
        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        self.list = QListWidget(self)
        self.layout.addWidget(self.list)
        self.list.setFixedWidth(210)

        # TODO option to hide all rigid stuff / do if non-rigid is there / just
        # always hide
        # TODO should i color stuff that has accepted CNMF output, or just hide
        # it?
        for d in self.motion_corrected_tifs:
            x = d.split(sep)
            fname_parts = x[-1].split('_')
            cor_type = 'rigid' if fname_parts[-1][:-4] == 'rig' else 'non-rigid'

            date_str = x[-4]
            fly_str = x[-3]
            thorimage_id = '_'.join(fname_parts[:-1])
            item_parts = [date_str, fly_str, thorimage_id, cor_type]

            item = QListWidgetItem('/'.join(item_parts))

            # TODO if add support for labelling single comparisons, check all
            # comparisons are added, and only color yellow if not all in db
            # TODO check whether source video was rigid / non-rigid, if going to
            # display both types of movies in list?
            analyzed = (
                (presentations.prep_date == date_str) &
                (presentations.fly_num == int(fly_str)) &
                (presentations.thorimage_id == thorimage_id)
            ).any()

            if analyzed:
                # TODO below, get item and set bg color if accepted
                item.setBackground(QColor('#7fc97f'))

            self.list.addItem(item)
            # TODO make make tooltip the full path? or show full path somewhere
            # else?

        self.list.itemDoubleClicked.connect(self.open_recording)

        # Other groups are: motion, online
        # TODO worth including patch_params?
        cnmf_groups = (
            'data',
            'patch_params',
            'preprocess_params',
            'init_params',
            'spatial_params',
            'temporal_params',
            'merging',
            'quality'
        )
        dont_show_by_group = defaultdict(set)
        # TODO get rid of most of this (maybe whole data tab if decay_time
        # moved to temporal) after loading from thorlabs metadata
        dont_show_by_group.update({
            'data': {'fnames','dims'}, #'fr','dxy'},
            'merging': {'gSig_range'},
        })

        # TODO TODO maybe make either just parameters or both params and data
        # explorer collapsible (& display intermediate results horizontally,
        # maybe only in this case?)
            
        # TODO or should this get a layout as input?
        cnmf_ctrl_widget = QWidget(self)
        self.layout.addWidget(cnmf_ctrl_widget)

        cnmf_ctrl_layout = QVBoxLayout(cnmf_ctrl_widget)
        cnmf_ctrl_widget.setLayout(cnmf_ctrl_layout)

        param_tabs = QTabWidget(cnmf_ctrl_widget)
        cnmf_ctrl_layout.addWidget(param_tabs)

        # TODO TODO eliminate vertical space between these rows of buttons
        shared_btns = QWidget(cnmf_ctrl_widget)
        cnmf_ctrl_layout.addWidget(shared_btns)
        shared_btn_layout = QVBoxLayout(shared_btns)
        shared_btns.setLayout(shared_btn_layout)
        shared_btn_layout.setSpacing(0)
        shared_btn_layout.setContentsMargins(0, 0, 0, 0)
        shared_btns.setFixedHeight(80)

        param_btns = QWidget(shared_btns)
        shared_btn_layout.addWidget(param_btns)
        param_btns_layout = QHBoxLayout(param_btns)
        param_btns.setLayout(param_btns_layout)
        param_btns_layout.setContentsMargins(0, 0, 0, 0)

        other_btns = QWidget(shared_btns)
        shared_btn_layout.addWidget(other_btns)
        other_btns_layout = QHBoxLayout(other_btns)
        other_btns.setLayout(other_btns_layout)
        other_btns_layout.setContentsMargins(0, 0, 0, 0)

        # TODO also give parent?
        reset_cnmf_params_btn = QPushButton('Reset All Parameters')
        reset_cnmf_params_btn.setEnabled(False)
        param_btns_layout.addWidget(reset_cnmf_params_btn)

        mk_default_params_btn = QPushButton('Make Parameters Default')
        #mk_default_params_btn.setEnabled(False)
        param_btns_layout.addWidget(mk_default_params_btn)
        mk_default_params_btn.clicked.connect(self.save_default_params)

        # TODO support this?
        # TODO TODO at least make not-yet-implemented stuff unclickable
        # TODO would need a format for these params... json?
        load_params_params_btn = QPushButton('Load Parameters From File')
        load_params_params_btn.setEnabled(False)
        param_btns_layout.addWidget(load_params_params_btn)

        save_params_btn = QPushButton('Save Parameters To File')
        save_params_btn.setEnabled(False)
        param_btns_layout.addWidget(save_params_btn)

        self.run_cnmf_btn = QPushButton('Run CNMF')
        other_btns_layout.addWidget(self.run_cnmf_btn)
        # Will enable after some data is selected. Can't run CNMF without that.
        self.run_cnmf_btn.setEnabled(False)

        save_cnmf_output_btn = QPushButton('Save CNMF Output')
        other_btns_layout.addWidget(save_cnmf_output_btn)
        save_cnmf_output_btn.setEnabled(False)

        self.run_cnmf_btn.clicked.connect(self.start_cnmf_worker)

        # TODO should save cnmf output be a button here or to the right?
        # maybe run should also be to the right?
        # TODO maybe a checkbox to save by default or something?

        # TODO maybe include a progress bar here? to the side? below?

        # TODO why is data/decay_time not in temporal parameters?
        self.default_json_params = '.default_cnmf_params.json'
        # TODO maybe search indep of cwd?
        if exists(self.default_json_params):
            print('Loading default parameters from {}'.format(
                self.default_json_params))

            self.params = params.CNMFParams.from_json(self.default_json_params)
        else:
            self.params = params.CNMFParams()

        # TODO TODO copy once here to get params to reset back to in GUI

        param_dict = self.params.to_dict()

        n = int(np.ceil(np.sqrt(len(cnmf_groups))))
        seen_types = set()
        formgen_print = False
        for i, g in enumerate(cnmf_groups):
            dont_show = dont_show_by_group[g]

            y = i % n
            x = i // n

            group_name = g.split('_')[0].title()
            if group_name == 'Init':
                group_name = 'Initialization'

            # TODO maybe there is no point in having groupbox actually?
            ##group = QGroupBox(group_name)
            ##param_layout.addWidget(group, x, y)

            tab = QWidget(param_tabs)
            param_tabs.addTab(tab, group_name)
            tab_layout = QVBoxLayout(tab)
            tab.setLayout(tab_layout)

            group = QWidget(tab)
            group_layout = QFormLayout(group)
            group.setLayout(group_layout)
            tab_layout.addWidget(group)

            if g == 'data':
                self.data_group_layout = group_layout

            reset_tab = QPushButton('Reset {} Parameters'.format(group_name))
            reset_tab.setEnabled(False)
            tab_layout.addWidget(reset_tab)
            # TODO allow setting defaults (in gui) that will override builtin
            # cnmf defaults, and persist across sessions
            # TODO do that w/ another button here?

            # TODO TODO TODO provide a "reset to defaults" option
            # maybe both within a tab and across all?
            # TODO maybe also a save all option? (if implementing that way...)

            # TODO TODO should patch / other params have a checkbox to disable
            # all params (there's a mode where everything is computed w/o
            # patches, right?)

            # TODO TODO blacklist approp parts of data / other things that can
            # (and should only) be filled in from thorimage metadata

            if formgen_print:
                print(g)

            # TODO maybe infer whether params are optional (via type
            # annotations, docstring, None already, or explicitly specifying)
            # and then have a checkbox to the side to enable whatever other
            # input there is?
            # TODO some mechanism for mutually exclusive groups of parameters
            # / parameters that force values of others
            for k, v in param_dict[g].items():
                if k in dont_show:
                    # maybe still print or something?
                    continue

                # CNMF has slightly different key names in the dict and object
                # representations of a set of params...
                # TODO maybe fix in CNMF in the future
                group_key = g.split('_')[0]

                # TODO maybe do this some other way / don't?
                # TODO make tooltip for each from docstring?
                # TODO TODO also get next line(s) for tooltip?

                doc_line = None
                for line in params.CNMFParams.__init__.__doc__.split('\n'):
                    if k + ':' in line:
                        doc_line = line.strip()
                        break

                print_stuff = False

                if doc_line is None:
                    warnings.warn(('parameter {} not defined in docstring'
                        ).format(k))
                    print_stuff = True
                # TODO also warn about params in docstring but not here
                # maybe parse params in docstring once first?

                # TODO how to handle stuff w/ None as value?
                # TODO how to handle stuff whose default is int but can be
                # float? just change default to float in cnmf?

                # TODO is underscore not displaying correctly? just getting cut
                # off? fix that? translate explicitely to a qlabel to maybe set
                # other properties about display?

                seen_types.add(str(type(v)))

                if type(v) is bool:
                    # TODO tristate in some cases?
                    w = QCheckBox(group)
                    w.setChecked(v)
                    assert not w.isTristate()

                    w.stateChanged.connect(
                        partial(self.set_boolean, group_key, k))

                elif type(v) is int:
                    # TODO set step relative to magnitude?
                    # TODO range?
                    w = QSpinBox(group)

                    int_min = -1
                    int_max = 10000
                    w.setRange(int_min, int_max)
                    assert v >= int_min and v <= int_max
                    w.setValue(v)

                    print_stuff = True
                    w.valueChanged.connect(
                        partial(self.set_from_spinbox, group_key, k))

                elif type(v) is float:
                    # TODO set step and decimal relative to default size?
                    # (1/10%?)
                    # TODO range?
                    # TODO maybe assume stuff in [0,1] should stay there?
                    w = QDoubleSpinBox(group)

                    float_min = -1.
                    float_max = 10000.
                    w.setRange(float_min, float_max)
                    assert v >= float_min and v <= float_max
                    w.setValue(v)

                    w.valueChanged.connect(
                        partial(self.set_from_spinbox, group_key, k))

                # TODO TODO if type is list (tuple?) try recursively looking up
                # types? (or just handle numbers?) -> place in
                # qwidget->qhboxlayout?

                elif type(v) is str:
                    # TODO use radio instead?
                    w = QComboBox(group)
                    w.addItem(v)

                    # TODO maybe should use regex?
                    v_opts = ([x.strip(" '") for x in doc_line.split(',')[0
                        ].split(':')[1].split('|')])

                    if formgen_print:
                        print('key:', k)
                        print('parsed options:', v_opts)
                    assert v in v_opts

                    for vi in v_opts:
                        if vi == v:
                            continue
                        w.addItem(vi)

                    print_stuff = True

                    w.currentIndexChanged[str].connect(
                        partial(self.set_from_list, group_key, k))

                else:
                    print_stuff = True
                    w = QLineEdit(group)
                    w.setText(repr(v))
                    # TODO TODO if using eval, use setValidator to set some
                    # validator that eval call works?
                    w.editingFinished.connect(
                        partial(self.set_from_text, group_key, k, w))

                if formgen_print and print_stuff:
                    print(k, v, type(v))
                    print(doc_line)
                    print('')

                group_layout.addRow(k, w)

            if formgen_print:
                print('')

        if formgen_print:
            print('Seen types:', seen_types)

        # TODO set tab index to most likely to change? spatial?

        # TODO TODO is there really any point to displaying anything here?
        # almost seems like button->dialog or something would be more
        # appropriate if not...
        # does the division make sense b/c some validation takes longer than
        # others (and that is what can be in the separate validation tab)?

        self.display_widget = QWidget(self)
        self.layout.addWidget(self.display_widget)
        self.display_layout = QVBoxLayout(self.display_widget)
        self.display_widget.setLayout(self.display_layout)
        # TODO TODO add toolbar / make image navigable

        self.plot_intermediates = True
        ########self.plot_intermediates = False
        self.fig = Figure()
        self.mpl_canvas = FigureCanvas(self.fig)
        self.display_layout.addWidget(self.mpl_canvas)

        nav_bar = NavigationToolbar(self.mpl_canvas, self)
        nav_bar.setFixedHeight(80)
        self.display_layout.addWidget(nav_bar)

        # TODO TODO accept / reject dialog beneath this
        # (maybe move save button out of ctrl widget? move reject in there?)
        display_btns = QWidget(self.display_widget)
        self.display_layout.addWidget(display_btns)
        display_btns.setFixedHeight(100)
        display_btns_layout = QHBoxLayout(display_btns)
        display_btns.setLayout(display_btns_layout)

        self.accept_cnmf_btn = QPushButton('Accept', display_btns)
        display_btns_layout.addWidget(self.accept_cnmf_btn)
        self.accept_cnmf_btn.clicked.connect(self.accept_cnmf)

        self.reject_cnmf_btn = QPushButton('Reject', display_btns)
        display_btns_layout.addWidget(self.reject_cnmf_btn)
        self.reject_cnmf_btn.clicked.connect(self.reject_cnmf)

        # TODO TODO warn if would run analysis on same data w/ same params as
        # had previously led to a rejection

        # TODO TODO TODO provide the opportunity to compare outputs of sets of
        # parameters, either w/ same displays side by side, or overlayed?

        # TODO TODO maybe some button to automatically pick best set of
        # parameters from database? (just max / some kind of average?)

        # TODO maybe share this across all widget classes?
        self.threadpool = QThreadPool()


    # TODO after implementing per-type, see if can be condensed to one function
    # for all types
    def set_boolean(self, group, key, qt_value):
        if qt_value == 0:
            new_value = False
        elif qt_value == 2:
            new_value = True
        else:
            raise ValueError('unexpected checkbox signal output')
        #print('Group:', group, 'Key:', key)
        #print('Old value:', self.params.get(group, key))
        self.params.set(group, {key: new_value})
        #print('New value:', self.params.get(group, key))


    # TODO so looks like this can be collapsed w/ list no problem? new name?
    # TODO wrap all these callbacks to enable/disable verbose stuff in one
    # place?
    def set_from_spinbox(self, group, key, new_value):
        #print('Group:', group, 'Key:', key)
        #print('Old value:', self.params.get(group, key))
        self.params.set(group, {key: new_value})
        #print('New value:', self.params.get(group, key))


    def set_from_list(self, group, key, new_value):
        #print('Group:', group, 'Key:', key)
        #print('Old value:', self.params.get(group, key))
        self.params.set(group, {key: new_value})
        #print('New value:', self.params.get(group, key))


    def set_from_text(self, group, key, qt_line_edit):
        if qt_line_edit.isModified():
            new_text = qt_line_edit.text()
            #print('new_text:', new_text)
            # TODO equivalent of ast.literal_eval that also works w/ things like
            # numpy arrays?
            # TODO TODO handle arrays. since their repr doesn't include prefix,
            # need to either import array here explicitly or detect and add
            # prefix
            new_value = eval(new_text)
            '''
            print('new_value:', new_value)
            print('repr(new_value):', repr(new_value))
            print('type(new_value):', type(new_value))
            '''
            self.params.set(group, {key: new_value})
            qt_line_edit.setModified(False)


    def start_cnmf_worker(self):
        self.run_cnmf_btn.setEnabled(False)
        self.accept_cnmf_btn.setEnabled(False)
        self.reject_cnmf_btn.setEnabled(False)
        # TODO separate button to cancel? change run-button to cancel?

        # TODO this delete all subplots, right?
        self.fig.clear()

        # TODO what kind of (if any) limitations are there on the extent to
        # which data can be shared across threads? can the worker modify
        # properties under self, and have those changes reflected here?

        # Pass the function to execute
        # Any other args, kwargs are passed to the run function
        worker = Worker(self.run_cnmf)
        # TODO so how does it know to pass one arg in this case?
        # (as opposed to cnmf_done case)
        worker.signals.result.connect(self.get_cnmf_output)
        worker.signals.finished.connect(self.cnmf_done)
        # TODO TODO implement. may require allowing callbacks to be passed into
        # cnmf code to report progress?
        ####worker.signals.progress.connect(self.progress_fn)

        self.parameter_json = self.params.to_json()
        self.cnmf_start = time.time()
        # Execute
        self.threadpool.start(worker)


    def run_cnmf(self):
        # TODO time cnmf run + report how long it took
        print('running cnmf')
        # TODO use cm.cluster.setup_cluster?
        # TODO what is the dview that returns as second arg?

        # TODO and maybe go further than just using deepcopy, to the extent that
        # deepcopy doesn't fully succeed in returning an object independent of
        # the original
        # Copying the parameters defensively, because CNMF has the bad habit of
        # changing the parameter object internally.
        self.params_copy = deepcopy(self.params)
        # TODO check / test that eq is actually working correctly
        assert self.params_copy == self.params

        # TODO i feel like this should be written to not require n_processes...
        n_processes = 1
        self.cnm = cnmf.CNMF(n_processes, params=self.params_copy)

        err_if_cnmf_changes_params = False
        if err_if_cnmf_changes_params:
            assert (self.params_copy == self.params,
                'CNMF changed params on init')

        # images : mapped np.ndarray of shape (t,x,y[,z])
        # TODO does it really need to be mapped? how can you even check if a
        # numpy array is mapped?
        # TODO maybe use fit_file for certain ways of getting here in the gui?
        # TODO TODO TODO check dims / C/F order
        self.cnm.fit(self.movie)

        # TODO see which parameters are changed?
        if err_if_cnmf_changes_params:
            assert self.params_copy == self.params, 'CNMF changed params in fit'

        # TODO maybe have a widget that shows the text output from cnmf?


    def cnmf_done(self):
        self.run_cnmf_btn.setEnabled(True)
        self.accept_cnmf_btn.setEnabled(True)
        self.reject_cnmf_btn.setEnabled(True)
        # TODO logging instead?
        print('done with cnmf')
        print('CNMF took {:.1f}s'.format(time.time() - self.cnmf_start))


    # TODO TODO actually provide a way to only initialize cnmf, to test out
    # various initialization procedures (though only_init arg doesn't actually
    # seem to accomplish this correctly)

    def get_cnmf_output(self):
        # TODO TODO allow toggling between type of background image shown
        # (radio / combobox for avg, snr, etc? "local correlations"?)
        # TODO TODO use histogram equalized avg image as one option
        img = self.avg

        only_init = self.params_copy.get('patch', 'only_init')

        n_axes = 4 if self.plot_intermediates and not only_init else 1
        contour_axes = self.fig.subplots(n_axes, 1, squeeze=False, sharex=True,
            sharey=True)

        self.fig.subplots_adjust(hspace=0, wspace=0)

        for i in range(n_axes):
            contour_ax = contour_axes[i, 0]
            contour_ax.axis('off')

            # TODO TODO make callbacks for each step and plot as they become
            # available
            if self.plot_intermediates:
                # TODO need to correct coordinates b/c slicing? just slice img?
                # TODO need to transpose or change C/F order or anything?
                if i == 0:
                    # TODO why are title's not working? need axis='on'?
                    # just turn off other parts of axis?
                    contour_ax.set_title('Initialization')
                    A = self.cnm.A_init
                elif i == 1:
                    contour_ax.set_title('After spatial update')
                    A = self.cnm.A_spatial_update_k[0]
                elif i == 2:
                    contour_ax.set_title('After merging')
                    A = self.cnm.A_after_merge_k[0]
                #elif i == 3:
                #    A = self.cnm.A_spatial_refinement_k[0]

            # TODO maybe show self.cnm.A_spatial_refinement_k[0] too in
            # plot_intermediates case? should be same though (though maybe one
            # is put back in original, non-sliced, coordinates?)
            if i == n_axes - 1 and not only_init:
                contour_ax.set_title('Final estimate')
                A = self.cnm.estimates.A

            caiman.utils.visualization.plot_contours(A, img, ax=contour_ax,
                display_numbers=False, colors='r', linewidth=1.0)

        # TODO maybe use this anyway, in case i am forgetting some other step
        # cnmf is doing?
        '''
        self.cnm.estimates.plot_contours(img=img, ax=contour_ax,
            display_numbers=False, colors='r', linewidth=1.0)
        '''

        self.fig.tight_layout()

        self.mpl_canvas.draw()
        # TODO maybe allow toggling same pane between avg and movie?
        # or separate pane for movie?
        # TODO use some non-movie version of pyqtgraph ImageView for avg,
        # to get intensity sliders? or other widget for that?


    def save_default_params(self):
        print('Writing new default parameters to {}'.format(
            self.default_json_params))
        # TODO TODO test round trip before terminating w/ success
        self.params.to_json(self.default_json_params)
        # TODO maybe use pickle for this?


    def common_run_info(self):
        global caiman_version_info
        run_info = {
            'run_at': [datetime.fromtimestamp(self.cnmf_start)],
            'input_filename': self.tiff_fname,
            'input_md5': self.tiff_md5,
            'input_mtime': self.tiff_mtime,
            'start_frame': self.start_frame,
            'stop_frame': self.stop_frame,
            'parameters': self.parameter_json,
            # TODO maybe share data / data browser similarly?
            'who': self.main_window.user,
            'host': socket.gethostname(),
            'host_user': getpass.getuser()
        }
        run_info.update(caiman_version_info)
        return run_info

    # TODO disable accept / reject buttons until after cnmf is run

    # TODO TODO TODO what happens if cnmf run then data is changed??
    # do accept / reject still (erroneously) work? should be sure to clear cnmf
    # output when changing data

    # TODO TODO add support for deleting presentations from db if reject
    # something that was just accepted?
    # TODO TODO dialog to confirm overwrite if accepting something already in
    # database?
    def accept_cnmf(self):
        # TODO delete me
        ACTUALLY_UPLOAD = True
        #

        # TODO maybe visually indicate which has been selected already?
        run_info = self.common_run_info()
        run_info['accepted'] = True
        run = pd.DataFrame(run_info)
        run.set_index('run_at', inplace=True)
        # TODO depending on what table is in method callable, may need to make
        # pd index match sql pk?
        # TODO test that result is same w/ or w/o method in case where row did
        # not exist, and that read shows insert worked in w/ method case
        if ACTUALLY_UPLOAD:
            run.to_sql('cnmf_runs', u.conn, if_exists='append',
                method=u.pg_upsert)

        # TODO just calculate metadata outright here?
            
        # TODO TODO save file to nas (particularly so that it can still be there
        # if database gets wiped out...) (should thus include parameters
        # [+version?] info)

        # TODO and refresh stuff in validation window s.t. this experiment now
        # shows up

        # TODO maybe also allow more direct passing of this data to other tab

        # x,y,n_footprints
        footprints = self.cnm.estimates.A.toarray()

        # Assuming equal number along both dimensions.
        pixels_per_side = int(np.sqrt(footprints.shape[0]))
        n_footprints = footprints.shape[1]

        footprints = np.reshape(footprints,
            (pixels_per_side, pixels_per_side, n_footprints))

        # frame number, cell -> value
        raw_f = self.cnm.estimates.C.T

        # TODO TODO to copy what Remy's matlab script does, need to detrend
        # within each "block"
        if self.cnm.estimates.F_dff is None:
            # quantileMin=8, frames_window=500, flag_auto=True, use_fast=False,
            # (a, b, C, f, YrA)
            self.cnm.estimates.detrend_df_f()

        df_over_f = self.cnm.estimates.F_dff.T

        # TODO TODO TODO just save a bunch of different versions of the df/f,
        # computed w/ extract / detrend, and any key changes in arguments, then
        # load that and plot some stuff for troubleshooting

        # TODO to test extract_DF_F, need Yr, A, C, bl
        # detrend_df_f wants A, b, C, f (YrA=None, but maybe it's used?
        # in this fn, they call YrA the "residual signals")
        sliced_movie = self.cnm.get_sliced_movie(self.movie)
        Yr = self.cnm.get_Yr(sliced_movie)
        ests = self.cnm.estimates

        '''
        try:
            # TODO fix save fn?
            # TODO test rt ser/deser first?
            self.cnm.save('dff_debug_cnmf.hdf5')
        except Exception as e:
            traceback.print_exc()
            print(e)
            import ipdb; ipdb.set_trace()
        '''
        
        footprint_dfs = []
        for cell_num in range(n_footprints):
            sparse = coo_matrix(footprints[:,:,cell_num])
            footprint_dfs.append(pd.DataFrame({
                'recording_from': [self.started_at],
                'cell': [cell_num],
                # Can be converted from lists of Python types, but apparently
                # not from numpy arrays or lists of numpy scalar types.
                # TODO check this doesn't transpose things
                # TODO TODO just move appropriate casting to my to_sql function,
                # and allow having numpy arrays (get type info from combination
                # of that and the database, like in other cases)
                'x_coords': [[int(x) for x in sparse.col.astype('int16')]],
                'y_coords': [[int(x) for x in sparse.row.astype('int16')]],
                'weights': [[float(x) for x in sparse.data.astype('float32')]]
            }))

        footprint_df = pd.concat(footprint_dfs, ignore_index=True)
        # TODO filter out footprints less than a certain # of pixels in cnmf?
        # (is 3 pixels really reasonable?)
        if ACTUALLY_UPLOAD:
            u.to_sql_with_duplicates(footprint_df, 'cells', verbose=True)

        # TODO store image w/ footprint overlayed?
        # TODO TODO maybe store an average frame of registered TIF, and then
        # indexes around that per footprint? (explicitly try to avoid responses
        # when doing so, for easier interpretation as a background?)

        # dims=dimensions of image (256x256)
        # T is # of timestamps

        # TODO why 474 x 4 + 548 in one case? i thought frame numbers were
        # supposed to be more similar... (w/ np.diff(odor_onset_frames))
        first_onset_frame_offset = \
            self.odor_onset_frames[0] - self.block_first_frames[0]

        n_frames, n_cells = df_over_f.shape
        assert n_cells == n_footprints

        start_frames = np.append(0,
            self.odor_onset_frames[1:] - first_onset_frame_offset)
        stop_frames = np.append(
            self.odor_onset_frames[1:] - first_onset_frame_offset - 1, n_frames)
        lens = [stop - start for start, stop in zip(start_frames, stop_frames)]

        # TODO delete version w/ int cast after checking they give same answers
        assert int(self.frame_times.shape[0]) == int(n_frames)
        assert self.frame_times.shape[0] == n_frames

        print(start_frames)
        print(stop_frames)
        # TODO find where the discrepancies are!
        print(sum(lens))
        print(n_frames)

        # TODO delete me
        # intended to use this to find best detrend / extract dff method
        try:
            state = {
                'Yr': Yr,
                'A': ests.A,
                'C': ests.C,
                'bl': ests.bl,
                'b': ests.b,
                'f': ests.f,
                'df_over_f': df_over_f,
                'start_frames': start_frames,
                'stop_frames': stop_frames,
                'date': self.date,
                'fly_num': self.fly_num,
                'thorimage_id': self.thorimage_id
            }
            with open('cnmf_state.p', 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            traceback.print_exc()
            print(e)
            import ipdb; ipdb.set_trace()
        #

        # TODO assert here that all frames add up / approx

        # TODO TODO either warn or err if len(start_frames) is !=
        # len(odor_pair_list)

        odor_id_pairs = [(o1,o2) for o1,o2 in
            zip(self.odor1_ids, self.odor2_ids)]
        print('odor_id_pairs:', odor_id_pairs)

        comparison_num = -1

        for i in range(len(start_frames)):
            if i % (self.presentations_per_repeat * self.n_repeats) == 0:
                comparison_num += 1
                repeat_nums = {id_pair: 0 for id_pair in odor_id_pairs}

            # TODO TODO also save to csv/flat binary/hdf5 per (date, fly,
            # thorimage)
            print('Processing presentation {}'.format(i))

            start_frame = start_frames[i]
            stop_frame = stop_frames[i]
            # TODO off by one?? check
            # TODO check against frames calculated directly from odor offset...
            # may not be const # frames between these "starts" and odor onset?
            onset_frame = start_frame + first_onset_frame_offset

            # TODO check again that these are always equal and delete
            # "direct_onset_frame" bit
            print('onset_frame:', onset_frame)
            direct_onset_frame = self.odor_onset_frames[i]
            print('direct_onset_frame:', direct_onset_frame)

            # TODO TODO why was i not using direct_onset_frame for this before?
            onset_time = self.frame_times[direct_onset_frame]
            assert start_frame < stop_frame
            # TODO check these don't jump around b/c discontinuities
            presentation_frametimes = \
                self.frame_times[start_frame:stop_frame] - onset_time
            # TODO delete try/except after fixing
            try:
                assert len(presentation_frametimes) > 1
            except AssertionError:
                print(self.frame_times)
                print(start_frame)
                print(stop_frame)
                import ipdb; ipdb.set_trace()

            # TODO TODO what caused the error here?
            odor_pair = odor_id_pairs[i]
            odor1, odor2 = odor_pair
            repeat_num = repeat_nums[odor_pair]
            repeat_nums[odor_pair] = repeat_num + 1

            offset_frame = self.odor_offset_frames[i]
            print('offset_frame:', offset_frame)
            assert offset_frame > direct_onset_frame
            # TODO share more of this w/ dataframe creation below, unless that
            # table is changed to just reference presentation table
            presentation = pd.DataFrame({
                'prep_date': [self.date],
                'fly_num': self.fly_num,
                'recording_from': self.started_at,
                'comparison': comparison_num,
                'odor1': odor1,
                'odor2': odor2,
                'repeat_num': repeat_num,
                'odor_onset_frame': direct_onset_frame,
                'odor_offset_frame': offset_frame,
                'from_onset': [[float(x) for x in presentation_frametimes]]
            })
            if ACTUALLY_UPLOAD:
                u.to_sql_with_duplicates(presentation, 'presentations')


            # maybe share w/ code that checks distinct to decide whether to
            # load / analyze?
            key_cols = [
                'prep_date',
                'fly_num',
                'recording_from',
                'comparison',
                'odor1',
                'odor2',
                'repeat_num'
            ]
            db_presentations = pd.read_sql('presentations', u.conn,
                columns=(key_cols + ['presentation_id']))

            presentation_ids = (db_presentations[key_cols] ==
                                presentation[key_cols].iloc[0]).all(axis=1)
            assert presentation_ids.sum() == 1
            presentation_id = db_presentations.loc[presentation_ids,
                'presentation_id'].iat[0]

            # TODO get remy to save it w/ less than 64 bits of precision?
            presentation_dff = df_over_f[start_frame:stop_frame, :]
            presentation_raw_f = raw_f[start_frame:stop_frame, :]

            # Assumes that cells are indexed same here as in footprints.
            cell_dfs = []
            for cell_num in range(n_cells):

                cell_dff = presentation_dff[:, cell_num].astype('float32')
                cell_raw_f = presentation_raw_f[:, cell_num].astype('float32')

                cell_dfs.append(pd.DataFrame({
                    'presentation_id': [presentation_id],
                    'recording_from': [self.started_at],
                    'cell': [cell_num],
                    'df_over_f': [[float(x) for x in cell_dff]],
                    'raw_f': [[float(x) for x in cell_raw_f]]
                }))
            response_df = pd.concat(cell_dfs, ignore_index=True)

            # TODO TODO test this is actually overwriting any old stuff
            if ACTUALLY_UPLOAD:
                u.to_sql_with_duplicates(response_df, 'responses')

            # TODO put behind flag
            db_presentations = pd.read_sql_query('SELECT DISTINCT prep_date, ' +
                'fly_num, recording_from, comparison FROM presentations',
                u.conn)

            print(db_presentations)
            print(len(db_presentations))
            #

            print('Done processing presentation {}'.format(i))

        # TODO check that all frames go somewhere and that frames aren't
        # given to two presentations. check they stay w/in block boundaries.
        # (they don't right now. fix!)

        self.current_item.setBackground(QColor('#7fc97f'))


    # TODO factor accept/reject into a label fn that takes accept as one boolean
    # arg
    def reject_cnmf(self):
        run_info = self.common_run_info()
        run_info['accepted'] = False
        run = pd.DataFrame(run_info)
        run.set_index('run_at', inplace=True)
        run.to_sql('cnmf_runs', u.conn, if_exists='append', method=u.pg_upsert)


    # TODO TODO either automatically considering change parameters / re-running
    # rejection or store parameters before labelling accept/reject

    # TODO maybe support save / loading cnmf state w/ their save/load fns w/
    # buttons in the gui? (maybe to some invisible cache?)

    def open_recording(self):
        idx = self.sender().currentRow()

        tiff = self.motion_corrected_tifs[idx]

        start = time.time()

        tiff_dir, tiff_just_fname = split(tiff)
        analysis_dir = split(tiff_dir)[0]
        full_date_dir, fly_dir = split(analysis_dir)
        date_dir = split(full_date_dir)[-1]

        date = datetime.strptime(date_dir, '%Y-%m-%d')
        fly_num = int(fly_dir)
        thorimage_id = tiff_just_fname[:4]

        # Trying all the operations that need to find files before setting any
        # instance variables, so that if those fail, we can stay on the current
        # data if we want (without having to reload it).
        ########################################################################
        # Start stuff more likely to fail (missing file, etc)
        ########################################################################

        mat = join(analysis_dir, rel_to_cnmf_mat, thorimage_id + '_cnmf.mat')

        try:
            evil.evalc("clear; data = load('{}', 'ti');".format(mat))
        except matlab.engine.MatlabExecutionError as e:
            # TODO inspect error somehow to see if it's a memory error?
            # -> continue if so
            # TODO print to stderr
            print(e)
            return
        ti = evil.eval('data.ti')

        recordings = df.loc[(df.date == date) &
                            (df.fly_num == fly_num) &
                            (df.thorimage_dir == thorimage_id)]
        recording = recordings.iloc[0]

        stimfile = recording['stimulus_data_file']
        stimfile_path = join(stimfile_root, stimfile)
        # TODO also err if not readable
        if not os.path.exists(stimfile_path):
            raise ValueError('copy missing stimfile {} to {}'.format(stimfile,
                stimfile_root))

        with open(stimfile_path, 'rb') as f:
            data = pickle.load(f)

        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.'.format(
                recording.project))
            return

        raw_fly_dir = join(raw_data_root, date_dir, fly_dir)
        thorsync_dir = join(raw_fly_dir, recording['thorsync_dir'])
        thorimage_dir = join(raw_fly_dir, recording['thorimage_dir'])
        stimulus_data_path = join(stimfile_root,
                                  recording['stimulus_data_file'])

        thorimage_xml_path = join(thorimage_dir, 'Experiment.xml')
        xml_root = etree.parse(thorimage_xml_path).getroot()

        data_params = cnmf_metadata_from_thor(tiff)

        started_at = \
            datetime.fromtimestamp(float(xml_root.find('Date').attrib['uTime']))

        # TODO see part of populate_db.py in this section to see how data
        # explorer list elements might be colored to indicate they have already
        # been run? or just get everything w/ pandas and color all from there?
        # TODO pane to show previous analysis runs of currently selected
        # experiment, or not worth it since maybe only want one accepted per?

        recordings = pd.DataFrame({
            'started_at': [started_at],
            'thorsync_path': [thorsync_dir],
            'thorimage_path': [thorimage_dir],
            'stimulus_data_path': [stimulus_data_path]
        })
        # TODO maybe defer this to accepting?
        u.to_sql_with_duplicates(recordings, 'recordings')

        n_repeats = int(data['n_repeats'])

        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors project.
        presentations_per_repeat = 3

        presentations_per_block = n_repeats * presentations_per_repeat

        n_blocks = int(len(data['odor_pair_list']) / presentations_per_block)

        # TODO TODO subset odor order information by start/end block cols
        # (for natural_odors stuff)
        # TODO TODO TODO also subset movie!! don't want partial blocks /
        # erroneous blocks included
        # TODO TODO and also subset by which frames are actually in ti
        # that alone will exclude partial blocks
        if pd.isnull(recording['first_block']):
            first_block = 0
        else:
            first_block = int(recording['first_block']) - 1

        if pd.isnull(recording['last_block']):
            last_block = n_blocks - 1
        else:
            last_block = int(recording['last_block']) - 1

        first_presentation = first_block * presentations_per_block
        last_presentation = (last_block + 1) * presentations_per_block

        odors = pd.DataFrame({
            'name': data['odors'],
            'log10_conc_vv': [0 if x == 'paraffin' else
                natural_odors_concentrations.at[x,
                'log10_vial_volume_fraction'] for x in data['odors']]
        })
        u.to_sql_with_duplicates(odors, 'odors')

        # TODO make unique id before insertion? some way that wouldn't require
        # the IDs, but would create similar tables?

        db_odors = pd.read_sql('odors', u.conn)
        # TODO TODO in general, the name alone won't be unique, so use another
        # strategy
        db_odors.set_index('name', inplace=True)

        # TODO test slicing
        # TODO make sure if there are extra trials in matlab, these get assigned
        # to first
        # + if there are less in matlab, should error
        odor_pair_list = \
            data['odor_pair_list'][first_presentation:last_presentation]

        assert (len(odor_pair_list) %
            (presentations_per_repeat * n_repeats) == 0)

        # TODO invert to check
        # TODO is this sql table worth anything if both keys actually need to be
        # referenced later anyway?

        # TODO only add as many as there were blocks from thorsync timing info?
        odor1_ids = [db_odors.at[o1,'odor'] for o1, _ in odor_pair_list]
        odor2_ids = [db_odors.at[o2,'odor'] for _, o2 in odor_pair_list]

        # TODO TODO make unique first. only need order for filling in the values
        # in responses.
        mixtures = pd.DataFrame({
            'odor1': odor1_ids,
            'odor2': odor2_ids
        })
        # TODO TODO maybe defer this to accepting...
        u.to_sql_with_duplicates(mixtures, 'mixtures')

        frame_times = np.array(ti['frame_times']).flatten()

        # Frame indices for CNMF output.
        # Of length equal to number of blocks. Each element is the frame
        # index (from 1) in CNMF output that starts the block, where
        # block is defined as a period of continuous acquisition.
        block_first_frames = np.array(ti['trial_start'], dtype=np.uint32
            ).flatten() - 1

        # stim_on is a number as above, but for the frame of the odor
        # onset.
        # TODO how does rounding work here? closest frame? first after?
        # TODO TODO did Remy change these variables? (i mean, it worked w/ some
        # videos?)
        end = time.time()
        print('Loading metadata took {:.3f} seconds'.format(end - start))

        # TODO TODO any way to only del existing movie if required to have
        # enough memory to load the new one?
        print('Loading tiff {}...'.format(tiff), end='', flush=True)
        start = time.time()
        # TODO is cnmf expecting float to be in range [0,1], like skimage?
        movie = tifffile.imread(tiff).astype('float32')
        '''
        try:
            movie = tifffile.imread(tiff).astype('float32')
        # TODO what is appropriate tifffile memory error to catch?
        except:
            # This strategy isn't ideal because it requires attempting the load
            # twice.
            # TODO if going to be doing this, need to make sure other data is
            # all cleared + cnm, regardless of whether other loads are
            # successful
            # TODO may need to check before attempting del (if on first run)
            # (or does del just do nothing if var isn't defined?)
            del self.movie
            movie = tifffile.imread(tiff).astype('float32')
        '''
        end = time.time()
        print(' done.')
        print('Loading TIFF took {:.3f} seconds'.format(end - start))

        ########################################################################
        # End stuff more likely to fail
        ########################################################################
        # Need to make sure we don't think the output of CNMF from other data is
        # associated with the new data we load.
        del self.cnm

        self.movie = movie

        self.current_item = self.sender().currentItem()
        self.date = date
        self.fly_num = fly_num
        self.thorimage_id = thorimage_id
        self.started_at = started_at
        self.n_repeats = n_repeats
        self.presentations_per_repeat = presentations_per_repeat
        self.odor1_ids = odor1_ids
        self.odor2_ids = odor2_ids
        self.frame_times = frame_times
        self.block_first_frames = block_first_frames

        self.odor_onset_frames = np.array(ti['stim_on'], dtype=np.uint32
            ).flatten() - 1
        self.odor_offset_frames = np.array(ti['stim_off'], dtype=np.uint32
            ).flatten() - 1

        # TODO TODO TODO if these are 1d, should be sorted... is Remy doing
        # something else weird?
        # (address after candidacy)

        # TODO set param as appropriate / maybe display in non-editable boxes in
        # data parameters
        # TODO how to refer to boxes by name, so the right ones can
        # automatically be set non-editable here?
        # TODO TODO should params just be set by programmatically changing box
        # and having that trigger callbacks??? or will that not trigger
        # callbacks?
        # TODO TODO rowCount vs count? itemAt seems to work off count
        # actually...
        for i in range(self.data_group_layout.count()):
            # TODO TODO does it always alternate qlabel / widget for that label?
            # would that ever break? otherwise, how to get a row, with the label
            # and the other widget for that row?
            item = self.data_group_layout.itemAt(i).widget()
            if type(item) == QLabel:
                continue

            label = self.data_group_layout.labelForField(item).text()
            if label not in data_params:
                continue

            v = data_params[label]

            # TODO maybe re-enable under some circumstances?
            item.setEnabled(False)

            # TODO maybe add support for the other types in case they change
            # data params
            if type(item) == QSpinBox or type(item) == QDoubleSpinBox:
                item.setValue(v)
            elif type(item) == QLineEdit:
                # TODO maybe make fn to change lineedit text, so repr/str could
                # be swapped in one spot
                # TODO maybe limit display of floating point numbers to a
                # certain number of decimal places?
                item.setText(repr(v))
                self.params.set('data', {label: v})

        # TODO worth setting any other parameters from thor metadata?
        '''
        # TODO just try to modify cnmf to work w/ uint16 as input?
        # might make it a bit faster...
        from_type = np.iinfo(self.movie.dtype)
        to_type = np.iinfo(np.dtype('float32'))

        to_type.max * (self.movie / from_type.max)
        '''
        # TODO maybe allow playing movie somewhere in display widget?

        self.avg = np.mean(self.movie, axis=0)
        self.tiff_fname = tiff

        start = time.time()
        # TODO maybe md5 array in memory, to not have to load twice?
        # (though for most formats it probably won't be the same... maybe none)
        self.tiff_md5 = md5(tiff)
        end = time.time()
        print('Hashing TIFF took {:.3f} seconds'.format(end - start))

        self.tiff_mtime = datetime.fromtimestamp(getmtime(tiff))

        self.start_frame = None
        self.stop_frame = None

        # want to keep this? (something like it, yes, but maybe replace w/
        # pyqtgraph video viewer roi, s.t. it can be restricted to a different
        # ROI if that would help)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(np.mean(self.movie, axis=(1,2)))
        self.mpl_canvas.draw()
        #

        self.run_cnmf_btn.setEnabled(True)


class ROIAnnotation(QWidget):
    # TODO though, if i'm going to require main_window to get some shared state,
    # and i don't have a fallback, might as well make it an explicit argument...
    def __init__(self, *args):
        super().__init__(*args)
        if len(args) == 1:
            self.main_window = args[0]

        self.setWindowTitle('ROI Validation')
        # TODO manage current user somehow? / work into database where
        # appropriate

        self.presentations = presentations.set_index(comp_cols)
        self.footprints = footprints

        # TODO TODO option to operate at recording level instead

        self.fly_comps = \
            presentations[comp_cols].drop_duplicates().sort_values(comp_cols)

        items = ['/'.join(r.split()[1:])
            for r in str(self.fly_comps).split('\n')[1:]]

        # TODO TODO should i share much of data selection code between other
        # widgets? make it a separate widget (and only one of them / one per
        # window? per window, probably?)?
        self.list = QListWidget(self)
        self.list.addItems(items)
        self.list.itemDoubleClicked.connect(self.open_comparison)

        # TODO or table w/ clickable rows? existing code for that?
        # TODO TODO could use QTableView
        # (see mfitzp/15-minute-apps currency example)

        # TODO maybe rename comparison according to mixture

        # TODO TODO maybe just make this the average image and overlay clickable
        # rois / make cells clickable?
        # TODO TODO TODO make a list widget part of this, but add stuff below to
        # use all cells / pick fraction and sample subset (seeded / not?) / 
        # maybe give letter labels + randomize order of real cell_ids, to the
        # extent that the way CNMF works causing some property of cells to vary
        # systematically along cell_id axis?
        self.cell_list = QListWidget(self)
        self.cell_list.itemDoubleClicked.connect(self.annotate_cell_comp)

        # TODO TODO maybe (pop up?) some kind of cell selector that lets you
        # pick a single cell to focus on? w/ diff display for / restriction to
        # those that have not yet been annotated?

        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        # this worked, but does above?
        #self.setLayout(self.layout)

        self.layout.addWidget(self.list)
        # TODO maybe use some space at bottom of list column for setting some
        # parameters like downsampling?
        # TODO maybe also # pixels surrounding ROIs to show
        self.list.setFixedWidth(160)

        self.layout.addWidget(self.cell_list)
        self.cell_list.setFixedWidth(50)

        # TODO TODO maybe make whole annotation layout thing its own widget?
        # or too useful to share state as we can now?

        # TODO should this have some explicit parent? layout or central_widget?
        self.annotation_widget = QWidget()
        self.layout.addWidget(self.annotation_widget)
        self.annotation_layout = QVBoxLayout(self.annotation_widget)
        self.annotation_widget.setLayout(self.annotation_layout)

        # TODO any benefit to specifying figsize? what's default?
        self.fig = Figure() #figsize=(6,2))
        self.mpl_canvas = FigureCanvas(self.fig)

        # TODO allow other numbers of repeats (/ diff layout to stack repeat?)
        self.axes = self.fig.subplots(1, 9)
        # TODO appropriate for a subplot? applies per-axis?
        #self.axes.hold(False)
        # TODO turn all axis off, etc
        for ax in self.axes.flat:
            ax.axis('off')

        # TODO maybe it's more important that this is proportional, for
        # non-fullscreen operation / diff screen size
        self.mpl_canvas.setFixedHeight(300)
        self.annotation_layout.addWidget(self.mpl_canvas)

        self.imv = pg.ImageView()

        # This approach will probably not work in future releases of pyqtgraph,
        # because the ROI handling in the Git master is different.
        # Works in 0.10.0
        # TODO TODO undo consequences of roi creation in ImageView construction
        # and then monkey patch in the ROI i want (others besides the below?)
        self.imv.getView().removeItem(self.imv.roi)
        # (getView().removeItem(...) for appropriate items[, etc?])
        # TODO use that roi to understand coord system?
        # TODO then call approp fns to get ROI to display
        # TODO not clear whether i also want to delete self.imv.normRoi...
        #import ipdb; ipdb.set_trace()

        self.imv.setImage(np.zeros((256, 256)))
        # TODO make image take up full space and have scroll go through
        # timesteps? or at least have scroll over time bar do that?

        # TODO should i have a separate view / horz concatenated movies to show
        # each trial side-by-side, or should i have one big movie?
        # TODO TODO maybe put average F for cell to the left of this?
        self.annotation_layout.addWidget(self.imv)
        # TODO make some marks to indicate when odors come on? (or something
        # like clickable links on the video navigation bar?)
        # TODO allow arbitrary movie playback speed, indep of downsampling

        self.label_button_widget = QWidget()
        self.label_button_widget.setFixedHeight(100)
        self.annotation_layout.addWidget(self.label_button_widget)
        self.label_button_layout = QHBoxLayout(self.label_button_widget)
        self.label_button_widget.setLayout(self.label_button_layout)
        # TODO some better way to get a set of buttons?
        # TODO should it be a radio anyway?
        for label in ('multiple_cells','not_full_cell','major_motion','good'):
            # TODO title first?
            button = QPushButton(label.replace('_', ' ').title(), \
                self.label_button_widget)
            # TODO need to keep a ~local reference as before?
            self.label_button_layout.addWidget(button)
            button.clicked.connect(self.label_footprint)

    
    # TODO TODO modify to work on either recordings or comparisons, and factor
    # s.t. all widgets can use it (or at least this and segmentation one)
    # TODO when refactoring, should i aim to share any loaded data across
    # widgets?
    def open_comparison(self):
        idx = self.sender().currentRow()
        comp = tuple(self.fly_comps.iloc[idx])
        self.fly_comp = comp

        # TODO rewrite to avoid performancewarning (indexing past lexsort depth)
        footprint_rows = self.footprints.loc[comp[:-1]]
        cells = footprint_rows.cell
        self.cell_list.clear()
        self.cell_list.addItems([str(c) for c in sorted(cells.to_list())])
        # TODO color list items to indicate already labelled
        # (including those from previous sessions in db)
        # red=not stable, orange=multiple cells, yellow=not all of the cell,
        # green=good on those three criteria
        # TODO other markers for "not a cell" / "too small" or something?
        # others? certainty (or just don't label uncertain ones?)?

        # TODO rewrite to avoid performancewarning (indexing past lexsort depth)
        self.metadata = self.presentations.loc[comp]

        tiff = motion_corrected_tiff_filename(*comp[:-1])
        # TODO move loading to some other process? QRunnable? progress bar?
        # TODO if not just going to load just comparison, keep movie loaded if
        # clicking other comparisons / until run out of memory
        # TODO just load part of movie for this comparison
        print('loading tiff {}...'.format(tiff), end='')
        self.movie = tifffile.imread(tiff)
        print(' done.')

        # Assumes all data between these frames is part of this comparison.
        self.first_frame = self.metadata.odor_onset_frame.min()
        self.last_frame = self.metadata.odor_offset_frame.max()
        # TODO assert no other onset / offset frames are in this range?

        print('movie.shape:', self.movie.shape)

        # TODO TODO could allow just comparison to be played... might be useful
        # (until a cell is clicked)

        # TODO maybe put the odors for this comparison in some text display
        # somewhere?

        # TODO maybe make response_calling_s a parameter modifiable in the GUI
        # TODO TODO put to left of where video is? left of annotation buttons?
        # under data selection part?
        response_calling_s = 3.0
        # TODO TODO TODO get fps from thor (check this works...)
        fps = fps_from_thor(self.metadata)
        self.fps = fps
        self.response_frames = int(np.ceil(fps * response_calling_s))
        self.background_frames = int(np.floor(fps * np.abs(
            self.metadata.from_onset.apply(lambda x: x.min()).min())))

        self.full_footprints = dict()
        for cell, data in footprint_rows[['cell','x_coords','y_coords','weights'
            ]].set_index('cell').iterrows():
            x_coords, y_coords, weights = data

            # TODO TODO TODO get this shape from thor metadata
            # TODO allow x and y diff?
            max_bound = 256
            footprint = np.array(coo_matrix((weights, (x_coords, y_coords)),
                shape=(max_bound, max_bound)).todense())

            self.full_footprints[cell] = footprint

        # TODO maybe put in some spatial structure to make range searches easier
        # first? i.e. bounding box overlap as at least a necessary condition for
        # roi overlap?
        # (could also save space on empty parts of matrices... how serious is
        # space consumption?)


    def annotate_cell_comp(self):
        # TODO do something else if not just in-order only
        # TODO TODO just use text converted to int...
        cell_id = self.sender().currentRow()

        (prep_date, fly_num, thorimage_id, comparison_num) = self.fly_comp
        # TODO probably worth also just showing the average F (as was used,
        # exclusively, before)?
        '''
        avg = cv2.normalize(avg, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        better_constrast = cv2.equalizeHist(avg)
        '''
        footprint = self.full_footprints[cell_id]

        # TODO TODO take the margin as a parameter in the GUI
        cropped_footprint, ((x_min, x_max), (y_min, y_max)) = \
            crop_to_nonzero(footprint, margin=6)

        near_footprints = dict()
        for c, f in self.full_footprints.items():
            if c == cell_id:
                continue

            # TODO maybe it should overlap by at least a certain number of
            # pixels? (i.e. pull back bounding box a little)
            in_display_region = f[x_min:x_max+1, y_min:y_max+1]
            if (in_display_region > 0).any():
                near_footprints[c] = in_display_region

        mpl_contour = None
        for ax, row in zip(self.axes.flat,
            self.metadata.sort_values('odor_onset_frame').itertuples()):

            odor_onset_frame = row.odor_onset_frame
            odor_offset_frame = row.odor_offset_frame
            curr_start_frame = odor_onset_frame - self.background_frames
            curr_stop_frame = odor_onset_frame + self.response_frames

            # TODO off by one?
            trial_background = self.movie[
                curr_start_frame:odor_onset_frame, x_min:x_max + 1,
                y_min:y_max + 1].mean(axis=0)

            # TODO divide by zero issues? how to avoid? suppress
            # warning if turn to NaN OK?
            # TODO TODO this order of operations is fine b/c
            # linearity of expectation, right?
            trial_dff = (self.movie[odor_onset_frame:curr_stop_frame,
                x_min:x_max + 1, y_min:y_max + 1].mean(axis=0)
                - trial_background) / trial_background

            # TODO maybe just set_data? and delete / clear contour artist?
            ax.clear()
            ax.axis('off')

            ax.imshow(trial_dff, cmap='gray')

            # TODO automatically loop through all cell ids and check for no
            # assertionerrors as a test / screening for test cases / bugs
            if mpl_contour is None:
                near_footprint_contours = dict()
                for c, f in near_footprints.items():
                    # TODO maybe indicate somehow visually if contour is cutoff
                    # (not?)hatched? don't draw as closed? diff color?
                    near_footprint_contours[c] = \
                        closed_mpl_contours(f, ax, colors='blue')

                contour = \
                    closed_mpl_contours(cropped_footprint, ax, colors='red')
            else:
                for c, cont in near_footprint_contours.items():
                    ax.plot(cont[:,0], cont[:,1], 'b-')

                ax.plot(contour[:,0], contour[:,1], 'r-')

            # TODO label trials above w/ odor (grouping options like kc_analysis
            # too?)

        self.mpl_canvas.draw()
        # TODO decrease border around edges of subplot grid

        #cropped_avg = \
        #    better_constrast[x_min:x_max + 1, y_min:y_max + 1]
        #ax.imshow(cropped_avg, cmap='gray')

        #######################################################################

        # TODO TODO parameters for border around footprint


        # TODO TODO TODO downsample!!! (or do w/ pg? seems it supports it?)

        self.imv.setImage(self.movie[self.first_frame:self.last_frame + 1,
            x_min:x_max + 1, y_min:y_max + 1])

        # TODO TODO how to keep roi on top, prevent it from being modified, and
        # clear them before next drawing?
        # TODO could probably use setMouseEnabled... to do the 2nd

        view_box = self.imv.getView()

        # TODO is just deleting the polylinerois sufficient to return the
        # viewbox close enough to its original state...? not clear

        to_remove = []
        for c in view_box.allChildren():
            if type(c) == pg.graphicsItems.ROI.PolyLineROI:
                to_remove.append(c)
        for r in to_remove:
            view_box.removeItem(r)

        for c, cont in near_footprint_contours.items():
            # TODO maybe allow clicking these to navigate to those cells?
            pg_roi = pg.PolyLineROI(cont, closed=True, pen='b', movable=False)
            # TODO make unmovable
            pg_roi.setZValue(20)
            view_box.addItem(pg_roi)

        # TODO since this works and polylineroi does not, maybe this is a bug in
        # imageview handling of ROIs s.t. it doesn't work w/ polylinerois?
        #pg_roi = pg.RectROI(pos=[0,0], size=10)

        # TODO isn't there some util fn to convert from numpy? use that?
        # TODO need to specify closed? doing what i expect?
        pg_roi = pg.PolyLineROI(contour, closed=True, pen='r', movable=False)

        # Just to copy what the 0.10.0 pyqtgraph does for it's default ROI.
        pg_roi.setZValue(20)

        # TODO TODO need to call view_box.invertY(True) as in RectROI
        # programcreek example ("upper left origin")? X?
        # so far, seems though inversion is not necessary
        # (assuming movie is not flipped wrt matplotlib images)
        # TODO TODO but movie may be... (seemingly along both axes?)

        # TODO TODO are bounds right? (some rois are getting clipped now)
        view_box.addItem(pg_roi)
        self.imv.roi = pg_roi

        # TODO should i explicitly disconnect previous roi before deleting, or
        # does it not matter / is that not a thing?
        pg_roi.sigRegionChanged.connect(self.imv.roiChanged)

        self.imv.roiChanged()

        # TODO TODO make it so ROI button only hides roiPlot, but not the roi
        # too

        # TODO simulate roiClicked w/ button checked? set button to checked in
        # init and maybe call roiClicked here?

        # TODO TODO how to make it loop?
        self.imv.play(self.fps)

        # TODO what is "A" icon that appears in bottom left of imv if you hover
        # there? (it changes axis, and i don't think want people to be able to
        # do that, particularly w/o some kind of reset button)
        # TODO something to allow user to collapse histogram

        # TODO TODO make pause / play button more prominent
        # TODO prevent time bar on bottom from having the range change (too
        # much) when clicking around in it


    def label_footprint(self):
        label = self.sender().text().replace(' ', '_').lower()
        print(label)
        if label == 'multiple_cells':
            values = {
                'only_one_cell': False,
                'all_of_cell': True,
                'stable': True
            }
        # TODO TODO maybe some way besides this to indicate multiple ROIs have
        # been assigned to one cell.
        # TODO allow changes (deletion / changing boundaries / creation) and
        # save them (might not be as useful for summary statistics on types of
        # errors? and may also not be that easy to act on to improve
        # algorithms? maybe it would be actionable...)
        # TODO maybe at least a separate 'duplicate' tag, for neighboring cells,
        # and one can be given not_full_cell label?
        # might be easier if you could also click neighboring rois in that case
        # , rather than scrolling through cell ids
        elif label == 'not_full_cell':
            values = {
                'only_one_cell': True,
                'all_of_cell': False,
                'stable': True
            }
        elif label == 'major_motion':
            values = {
                'only_one_cell': True,
                'all_of_cell': True,
                'stable': False
            }
        elif label == 'good':
            values = {
                'only_one_cell': True,
                'all_of_cell': True,
                'stable': True
            }
        # TODO some way to indicate missing cells

        # TODO combine w/ current cell/comparison info and...
        # TODO TODO insert values into database
        

    # TODO TODO TODO hotkeys / buttons for labelling

    # TODO TODO if doing cnmf in this gui, save output directly to database


# TODO TODO maybe make a "response calling" tab that does what my response
# calling / mpl annotation is doing?

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_widget.setLayout(central_layout)

        self.user_widget = QWidget(self)
        central_layout.addWidget(self.user_widget)

        self.user_widget.setFixedHeight(40)
        self.user_widget.setFixedWidth(170)

        user_layout = QHBoxLayout(self.user_widget)
        self.user_widget.setLayout(user_layout)
        user_label = QLabel('User', self.user_widget)
        # TODO maybe attribute on people as to whether to show here or not?
        user_select = QComboBox(self.user_widget)

        # TODO something like a <select user> default?
        # TODO TODO make user persist across restarts of the program
        # (w/o needing any special actions, i think)
        # TODO some kind of atexit pickling?
        # TODO (just add default selection first)
        self.user_cache_file = '.cnmf_gui_user.p'
        self.user = self.load_default_user()

        self.nicknames = set(pd.read_sql('people', u.conn)['nickname'])
        if self.user is not None:
            user_select.addItem(self.user)

        for nickname in self.nicknames:
            if nickname != self.user:
                user_select.addItem(nickname)

        user_select.setEditable(True)
        # TODO maybe turn off blinking cursor thing when this box isn't focused?
        # general display practices when using an editable combobox?
        user_select.currentIndexChanged[str].connect(self.change_user)
        # TODO maybe make this a cache w/ more generic name if i have need to
        # save other settings

        user_layout.addWidget(user_label)
        user_layout.addWidget(user_select)

        self.tabs = QTabWidget(central_widget)
        central_layout.addWidget(self.tabs)

        # TODO maybe an earlier tab for motion correction or something at some
        # point?
        self.mc_tab = MotionCorrection(self)
        self.seg_tab = Segmentation(self)
        self.validation_tab = ROIAnnotation(self)

        # TODO factor add + windowTitle bit to fn?
        self.tabs.addTab(self.mc_tab, self.mc_tab.windowTitle())
        seg_index = self.tabs.addTab(self.seg_tab, self.seg_tab.windowTitle())

        val_index = self.tabs.addTab(self.validation_tab,
            self.validation_tab.windowTitle())

        #self.tabs.setCurrentIndex(val_index)
        self.tabs.setCurrentIndex(seg_index)

        # i can't seem to get the space between combobox and tabs any smaller...
        central_layout.setSpacing(0)
        central_layout.setContentsMargins(0, 0, 0, 0)
        # (left, top, right, bottom)
        user_layout.setContentsMargins(10, 0, 10, 0)
        central_layout.setAlignment(self.user_widget, Qt.AlignLeft)

    
    def change_user(self, user):
        if user not in self.nicknames:
            pd.DataFrame({'nickname': [user]}).to_sql('people', u.conn,
                if_exists='append', index=False)
            self.nicknames.add(user)
        self.user = user


    def save_default_user(self):
        if self.user is None:
            return
        with open(self.user_cache_file, 'wb') as f:
            pickle.dump(self.user, f)


    def load_default_user(self):
        if not exists(self.user_cache_file):
            return None
        with open(self.user_cache_file, 'rb') as f:
            return pickle.load(f)


    def closeEvent(self, event):
        self.save_default_user()


def main():
    global odors
    global recordings
    global presentations
    global rec_meta_cols
    global recordings_meta
    global footprints
    global comp_cols
    global caiman_version_info
    global evil

    # Calling this first to minimize chances of code diverging.
    caiman_version_info = u.caiman_version_info()

    print('reading odors from postgres...', end='')
    odors = pd.read_sql('odors', u.conn)
    print(' done')

    print('reading presentations from postgres...', end='')
    presentations = pd.read_sql('presentations', u.conn)
    print(' done')

    presentations['from_onset'] = presentations.from_onset.apply(
        lambda x: np.array(x))

    presentations = merge_odors(presentations, odors)

    # TODO change sql for recordings table to use thorimage dir + date + fly
    # as index?
    recordings = pd.read_sql('recordings', u.conn)

    presentations = merge_recordings(presentations, recordings)

    rec_meta_cols = [
        'recording_from',
        'prep_date',
        'fly_num',
        'thorimage_id',
        'thorimage_path',
        'thorsync_path',
        'stimulus_data_path'
    ]
    recordings_meta = presentations[rec_meta_cols].drop_duplicates()

    footprints = pd.read_sql('cells', u.conn)
    footprints = footprints.merge(recordings_meta, on='recording_from')
    footprints.set_index(recording_cols, inplace=True)

    comp_cols = recording_cols + ['comparison']

    evil = u.matlab_engine()

    # TODO convention re: this vs setWindowTitle? latter not available if making
    # a window out of a widget?
    # TODO maybe setWindowTitle based on the tab? or add to end?
    app = QApplication(['2p analysis GUI'])
    win = MainWindow()
    win.showMaximized()
    app.exit(app.exec_())


if __name__ == '__main__':
    main()

