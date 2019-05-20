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
import hashlib
import time
from datetime import datetime
import socket
import getpass
import pickle
from copy import deepcopy
from io import BytesIO
import pprint
import traceback

from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot, QThreadPool, QRunnable,
    Qt, QEvent, QVariant)
from PyQt5.QtGui import QColor, QKeySequence, QCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QListWidget,
    QGroupBox, QPushButton, QLineEdit, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QLabel, QListWidgetItem, QScrollArea, QAction, QShortcut,
    QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QFileDialog, QMenu, QMessageBox)
import sip

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

# TODO allow things to fail gracefully if we don't have cnmf.
# either don't display tabs that involve it or display a message indicating it
# was not found.
import caiman
from caiman.source_extraction.cnmf import params, cnmf
import caiman.utils.visualization

import hong2p.util as u


# Maybe rename. It's these cols once already in a recording + comparison.
cell_cols = ['name1','name2','repeat_num','cell']

raw_data_root = u.raw_data_root()
analysis_output_root = u.analysis_output_root()
stimfile_root = u.stimfile_root()

use_cached_gsheet = False
show_inferred_paths = True
overwrite_older_analysis = True

df = u.mb_team_gsheet(use_cache=use_cached_gsheet)
#import ipdb; ipdb.set_trace()

rel_to_cnmf_mat = 'cnmf'

natural_odors_concentrations = pd.read_csv('natural_odor_panel_vial_concs.csv')
# TODO maybe assert no duplicate names first
natural_odors_concentrations.set_index('name', inplace=True)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# TODO worth allowing selection of a folder?
# TODO worth saving labels to some kind of file (CSV?) in case database not
# reachable? fn to load these to database?
# TODO alternative to database for input (for metadata mostly)?

# TODO TODO support loading single comparisons from tiffs in util

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
            self.signals.finished.emit()


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
        self.motion_corrected_tifs = u.list_motion_corrected_tifs()

        self.splitter = QSplitter(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.splitter)

        self.data_and_ctrl_widget = QWidget(self)
        self.splitter.addWidget(self.data_and_ctrl_widget)

        self.data_and_ctrl_layout = QHBoxLayout(self.data_and_ctrl_widget)
        # TODO try deleting all calls to setLayout. may work fine?
        # (as long as layout inits are passed the same widget)
        self.data_and_ctrl_widget.setLayout(self.data_and_ctrl_layout)
        self.data_and_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        # TODO TODO make a refresh button for this widget, which re-reads from
        # db (in case other programs have edited it)
        # TODO get rid of space on left that wasnt there w/ tree widget
        # TODO get rid of / change top label thing
        self.data_tree = QTreeWidget(self)
        self.data_tree.setHeaderHidden(True)
        self.data_and_ctrl_layout.addWidget(self.data_tree)
        self.data_tree.setFixedWidth(210)
        self.data_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_tree.customContextMenuRequested.connect(self.data_tree_menu)

        self.accepted_color = '#7fc97f'
        self.rejected_color = '#ff4d4d'

        global recordings
        for d in self.motion_corrected_tifs:
            x = d.split(sep)
            fname_parts = x[-1].split('_')
            # TODO maybe just get rid of last part?
            cor_type = 'rigid' if fname_parts[-1][:-4] == 'rig' else 'non-rigid'

            date_str = x[-4]
            fly_str = x[-3]
            # TODO make a fn for this, in case naming convention changes again?
            thorimage_id = '_'.join(fname_parts[:-1])
            item_parts = [date_str, fly_str, thorimage_id, cor_type]

            #item = QListWidgetItem('/'.join(item_parts))
            # TODO need setText call here?
            item = QTreeWidgetItem(self.data_tree)
            # 0 for 0th column
            item.setText(0, '/'.join(item_parts))
            self.data_tree.addTopLevelItem(item)

            tif_seg_runs = u.list_segmentations(d)
            if tif_seg_runs is None:
                # There cannot be a canonical_segmentation is db if 
                # there are no segmentation runs for this recording.
                item.setData(0, Qt.UserRole, pd.NaT)
                continue

            recording_from = tif_seg_runs.recording_from.unique()
            assert len(recording_from) == 1
            recording_from = recording_from[0]
            canonical_segmentation = recordings.set_index('started_at').at[
                recording_from, 'canonical_segmentation']
            item.setData(0, Qt.UserRole, canonical_segmentation)

            # TODO maybe replace this any_accepted stuff w/ call to
            # self.color_recording_node
            any_accepted = False
            for _, r in tif_seg_runs.iterrows():
                segrun = self.add_segrun_widget(item, r)

                if r.accepted == True:
                    any_accepted = True
                    if r.run_at == canonical_segmentation:
                        canonical = True
                    else:
                        canonical = False
                    self.mark_canonical(segrun, canonical)
                else:
                    self.mark_canonical(segrun, None)

            if any_accepted:
                item.setBackground(0, QColor(self.accepted_color))

        self.data_tree.itemDoubleClicked.connect(self.handle_treeitem_click)

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

            self.params = \
                params.CNMFParams.from_json_file(self.default_json_params)
        else:
            self.params = params.CNMFParams()

        # TODO TODO copy once here to get params to reset back to in GUI
        self.cnmf_ctrl_widget = self.make_cnmf_param_widget(self.params,
            editable=True)
        self.param_display_widget = None

        self.param_widget_stack = QStackedWidget(self)
        self.param_widget_stack.addWidget(self.cnmf_ctrl_widget)
        self.data_and_ctrl_layout.addWidget(self.param_widget_stack)

        self.display_widget = QWidget(self)
        self.splitter.addWidget(self.display_widget)


        self.splitter_handle = self.splitter.handle(1)
        self.splitter_bar_layout = QVBoxLayout()
        self.splitter_bar_layout.setContentsMargins(0, 0, 0, 0)
        # TODO make this button a little bigger/taller / easier to click?
        self.splitter_btn = QToolButton(self.splitter_handle)
        self.splitter_btn.clicked.connect(self.toggle_splitter)
        self.splitter_bar_layout.addWidget(self.splitter_btn)
        self.splitter_handle.setLayout(self.splitter_bar_layout)

        self.splitter_btn.setArrowType(Qt.LeftArrow)
        self.splitter_collapsed = False

        self.display_layout = QVBoxLayout(self.display_widget)
        self.display_widget.setLayout(self.display_layout)
        self.display_layout.setSpacing(0)

        # TODO put in some kind of GUI-settable persistent options?
        self.plot_intermediates = False
        self.fig = Figure()
        self.mpl_canvas = FigureCanvas(self.fig)
        # TODO this fixed_dpi thing work? let me make things resizable again?
        # unclear... probably no?
        ###self.mpl_canvas.fixed_dpi = 100
        
        # TODO test that zoom doesn't change what is serialized. especially
        # the png / svgs
        self.scrollable_canvas = QScrollArea(self)
        self.current_zoom = 1.0
        self.max_zoom = 5.0
        self.min_zoom = 0.1
        self.fig_w_inches = 7
        self.fig_h_inches = 7

        # TODO at least set horz scroll to center
        # TODO but maybe always load things zoomed out s.t. everything is
        # visible (self.current_zoom probably < 1)
        self.scrollable_canvas.setWidget(self.mpl_canvas)
        self.scrollable_canvas.setWidgetResizable(False)
        # This looks for the eventFilter function of this class.
        self.scrollable_canvas.viewport().installEventFilter(self)

        self.display_layout.addWidget(self.scrollable_canvas)

        self.nav_bar = NavigationToolbar(self.mpl_canvas, self)
        self.nav_bar.setFixedHeight(80)
        self.display_layout.addWidget(self.nav_bar)

        # TODO TODO TODO maybe make display_widget tabbed, w/ one tab as it
        # currently is, and the other to control postpressing params (like
        # response window length for correlations, trace extraction, etc) and
        # update the display on the former tab. should switch back to other tab
        # when pressing update.
        # (tabbed just b/c window is getting pretty crowded now...)
        # could maybe fit in at the bottom or something. collapsible?
        # would be nice to share parameter widget code with that for cnmf...
        # (break into fn that parses cnmf docstring and another that takes that
        # output and makes the widget)

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
        # TODO further decrease space between btns and checkboxes
        display_btns_layout.setContentsMargins(0, 0, 0, 0)

        display_params = QWidget(self.display_widget)
        #display_params.setFixedHeight(30)
        self.display_layout.addWidget(display_params)
        display_params_layout = QHBoxLayout(display_params)
        display_params.setLayout(display_params_layout)
        display_params_layout.setSpacing(0)
        display_params_layout.setContentsMargins(0, 0, 0, 0)

        # TODO TODO make either display params or them + buttons collapsible
        # w/ a vertical splitter
        # TODO TODO also include display params in default param json?
        # just save automatically but separately?
        self.plot_corrs_btn = QCheckBox('Correlations', display_params)
        self.plot_correlations = True
        self.plot_corrs_btn.setChecked(self.plot_correlations)
        self.plot_corrs_btn.stateChanged.connect(partial(
            self.set_boolean, 'plot_correlations'))
        display_params_layout.addWidget(self.plot_corrs_btn)

        self.plot_traces_btn = QCheckBox('Traces', display_params)
        self.plot_traces = False
        self.plot_traces_btn.setChecked(self.plot_traces)
        self.plot_traces_btn.stateChanged.connect(partial(
            self.set_boolean, 'plot_traces'))
        display_params_layout.addWidget(self.plot_traces_btn)

        # TODO TODO warn if would run analysis on same data w/ same params as
        # had previously led to a rejection

        # TODO TODO TODO provide the opportunity to compare outputs of sets of
        # parameters, either w/ same displays side by side, or overlayed?

        # TODO TODO maybe some button to automatically pick best set of
        # parameters from database? (just max / some kind of average?)

        # TODO maybe share this across all widget classes?
        self.threadpool = QThreadPool()

        self.accept_cnmf_btn.setEnabled(False)
        self.reject_cnmf_btn.setEnabled(False)
        self.movie = None
        self.cnm = None
        self.cnmf_running = False
        self.params_changed = False
        self.accepted = None

        # TODO delete / handle differently
        self.ACTUALLY_UPLOAD = True
        #


    # TODO TODO provide facilities to show which differ
    # from current set of params, at least. maybe between arbitrary selections
    # of pairs. maybe just color backgrounds of widgets for differing params 
    # yellow or something (and overlay current value?)
    # TODO TODO what happens in case where cnmf changes and a param is now:
    # 1) missing
    # 2) added
    # 3) of a different type
    #    - changing *_border to tuple broke this, as could still only enter an
    #      int
    # any of these cases fail?
    def make_cnmf_param_widget(self, cnmf_params, editable=False):
        """
        """
        # TODO get from output (in the one editable case) rather than setting in
        # here and checking at each start?
        if editable and hasattr(self, 'data_group_layout'):
            raise ValueError('can not make two editable widgets')

        # TODO TODO need to make sure that only one editable version can be
        # created at once
        # TODO TODO need to make sure that if non-editable are connected to save
        # param callback, they use the correct data
        cnmf_ctrl_widget = QWidget(self)
        param_tabs = QTabWidget(cnmf_ctrl_widget)

        # TODO could maybe take a dict of type (class) -> callback
        # and then pass self.set_from* fns in. might be tricky.
        # (to factor this out of class)

        param_json_str = None
        if type(cnmf_params) is str:
            param_json_str = cnmf_params
            cnmf_params = params.CNMFParams.from_json(cnmf_params)

        elif type(cnmf_params) is not params.CNMFParams:
            raise ValueError('cnmf_params must be of type CNMFParams or str')

        param_dict = cnmf_params.to_dict()

        # Other groups are: motion, online
        # TODO rename to indicate these are the subset we want visible in gui?
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
            'data': {'fnames','dims'},
            'merging': {'gSig_range'},
        })

        seen_types = set()
        formgen_print = False
        for i, g in enumerate(cnmf_groups):
            dont_show = dont_show_by_group[g]

            group_name = g.split('_')[0].title()
            if group_name == 'Init':
                group_name = 'Initialization'

            tab = QWidget(param_tabs)
            param_tabs.addTab(tab, group_name)
            tab_layout = QVBoxLayout(tab)
            tab.setLayout(tab_layout)

            group = QWidget(tab)
            group_layout = QFormLayout(group)
            group.setLayout(group_layout)
            tab_layout.addWidget(group)

            if editable:
                # TODO just get this from output somehow?
                if g == 'data':
                    self.data_group_layout = group_layout

                reset_tab = \
                    QPushButton('Reset {} Parameters'.format(group_name))
                reset_tab.setEnabled(False)
                tab_layout.addWidget(reset_tab)

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

                    if editable:
                        w.stateChanged.connect(
                            partial(self.cnmf_set_boolean, group_key, k))

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

                    if editable:
                        w.valueChanged.connect(
                            partial(self.cnmf_set_from_spinbox, group_key, k))

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

                    if editable:
                        w.valueChanged.connect(
                            partial(self.cnmf_set_from_spinbox, group_key, k))

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

                    if editable:
                        w.currentIndexChanged[str].connect(
                            partial(self.cnmf_set_from_list, group_key, k))

                else:
                    print_stuff = True
                    w = QLineEdit(group)
                    w.setText(repr(v))
                    if editable:
                        # TODO TODO if using eval, use setValidator to set some
                        # validator that eval call works?
                        w.editingFinished.connect(
                            partial(self.cnmf_set_from_text, group_key, k, w))

                if formgen_print and print_stuff:
                    print(k, v, type(v))
                    print(doc_line)
                    print('')

                if not editable:
                    w.setEnabled(False)

                group_layout.addRow(k, w)

            if formgen_print:
                print('')

        if formgen_print:
            print('Seen types:', seen_types)

        # TODO set tab index to most likely to change? spatial?

        cnmf_ctrl_layout = QVBoxLayout(cnmf_ctrl_widget)

        # TODO TODO how to replace this widget w/ new one if loading serialized
        # params into totally new QTabWidget??
        cnmf_ctrl_layout.addWidget(param_tabs)

        # TODO eliminate vertical space between these rows of buttons
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

        if editable:
            # TODO also give parent?
            reset_cnmf_params_btn = QPushButton('Reset All Parameters')
            reset_cnmf_params_btn.setEnabled(False)
            param_btns_layout.addWidget(reset_cnmf_params_btn)

        mk_default_params_btn = QPushButton('Make Parameters Default')
        #mk_default_params_btn.setEnabled(False)
        param_btns_layout.addWidget(mk_default_params_btn)
        # TODO TODO make this possible in non-editable data case
        # (probably store alongside self.params and pass extra arg to this fn as
        # here?)
        if editable:
            mk_default_params_btn.clicked.connect(self.save_default_params)
        else:
            assert param_json_str is not None
            # TODO TODO test this case
            mk_default_params_btn.clicked.connect(
                partial(self.save_default_params, param_json_str))

        if editable:
            # TODO support this?
            load_params_params_btn = QPushButton('Load Parameters From File')
            load_params_params_btn.setEnabled(False)
            param_btns_layout.addWidget(load_params_params_btn)

        save_params_btn = QPushButton('Save Parameters To File')
        if editable:
            save_params_btn.setEnabled(False)
        else:
            assert param_json_str is not None
            # TODO why didn't lambdas work before again? this case suffer from
            # the same problem?
            save_params_btn.clicked.connect(
                lambda: self.save_json(param_json_str))
            # this didn't work, seemingly b/c interaction w/ *args
            # (it'd pass in (False,) for args)
            #    partial(self.save_json, param_json_str))

        param_btns_layout.addWidget(save_params_btn)

        if editable:
            other_btns = QWidget(shared_btns)
            shared_btn_layout.addWidget(other_btns)
            other_btns_layout = QHBoxLayout(other_btns)
            other_btns.setLayout(other_btns_layout)
            other_btns_layout.setContentsMargins(0, 0, 0, 0)

            # TODO TODO allow this to be checked while data is loading, and then
            # just start as soon as data finishes loading
            self.run_cnmf_btn = QPushButton('Run CNMF')
            other_btns_layout.addWidget(self.run_cnmf_btn)
            # Will enable after some data is selected. Can't run CNMF without
            # that.
            self.run_cnmf_btn.setEnabled(False)

            save_cnmf_output_btn = QPushButton('Save CNMF Output')
            other_btns_layout.addWidget(save_cnmf_output_btn)
            save_cnmf_output_btn.setEnabled(False)

            self.run_cnmf_btn.clicked.connect(self.start_cnmf_worker)

        return cnmf_ctrl_widget


    def add_segrun_widget(self, parent, segrun_row) -> None:
        seg_run_item = QTreeWidgetItem(parent)
        seg_run_item.setData(0, Qt.UserRole, segrun_row)
        seg_run_item.setText(0, str(segrun_row.run_at)[:16])
        seg_run_item.setFlags(seg_run_item.flags() ^ Qt.ItemIsUserCheckable)

        if segrun_row.accepted == True:
            seg_run_item.setBackground(0, QColor(self.accepted_color))
        elif segrun_row.accepted == False:
            seg_run_item.setBackground(0, QColor(self.rejected_color))

        return seg_run_item


    def toggle_splitter(self):
        # TODO maybe go back to last position when uncollapsing
        # (and save position when collapsing)
        if self.splitter_collapsed:
            self.splitter_btn.setArrowType(Qt.LeftArrow)
            self.splitter.setSizes(self.last_splitter_sizes)
            self.splitter_collapsed = False
        else:
            # TODO set arrow dir (can this be changed / how?)
            self.splitter_btn.setArrowType(Qt.RightArrow)
            self.last_splitter_sizes = self.splitter.sizes()
            self.splitter.setSizes([0, 1])
            self.splitter_collapsed = True


    def delete_segrun(self, treenode) -> None:
        run_at = treenode.data(0, Qt.UserRole).run_at

        # TODO change order of yes / no?
        confirmation_choice = QMessageBox.question(self, 'Confirm delete',
            'Remove analysis run from {} from database?'.format(
            str(run_at)[:16]),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if confirmation_choice == QMessageBox.Yes:
            print('Deleting analysis run {} from database'.format(run_at))

            sql = "DELETE FROM analysis_runs WHERE run_at = '{}'".format(run_at)
            ret = u.conn.execute(sql)

            parent = treenode.parent()

            self.data_tree.removeItemWidget(treenode)
            # maybe remove this sip bit... not sure it's necessary
            sip.delete(treenode)
            # alternative would probably be:
            # treenode.deleteLater()
            # see: stackoverflow.com/questions/5899826

            self.color_recording_node(parent)


    def color_recording_node(self, treenode) -> None:
        if treenode.parent() is not None:
            self.color_recording_node(treenode.parent())

        any_accepted = False
        for i in range(treenode.childCount()):
            if treenode.child(i).data(0, Qt.UserRole).accepted:
                any_accepted = True
                break
                
        # TODO maybe color red if at least one thing is not accepted?
        if any_accepted:
            item.setBackground(0, QColor(self.accepted_color))
        else:
            # TODO test this
            # got from: stackoverflow.com/questions/37761002
            item.setData(0, Qt.BackgroundRole, None)


    def mark_canonical(self, treenode, canonical) -> None:
        if canonical == True:
            treenode.setCheckState(0, Qt.Checked)
        elif canonical == False:
            treenode.setCheckState(0, Qt.Unchecked)
        elif canonical is None:
            # For deleting checkbox of something that was once marked accepted
            # and is now rejected.
            treenode.setData(0, Qt.CheckStateRole, QVariant())


    def make_canonical(self, treenode) -> None:
        row = treenode.data(0, Qt.UserRole)

        old_canonical = treenode.parent().data(0, Qt.UserRole)
        if old_canonical == row.run_at:
            return

        if not pd.isnull(old_canonical):
            for i in range(parent.childCount()):
                if parent.child(i).data(0, Qt.UserRole).run_at == old_canonical:
                    self.mark_canonical(parent.child(i), False)
                    break

        sql = ("UPDATE recordings SET canonical_segmentation = " +
            "'{}' WHERE started_at = '{}'").format(row.run_at,
            row.recording_from)
        ret = u.conn.execute(sql)

        self.mark_canonical(treenode, True)


    def data_tree_menu(self, pos):
        node = self.data_tree.itemAt(pos)

        if node is None:
            return

        if node.parent() is not None:
            menu = QMenu(self)
            menu.addAction('Delete from database',
                lambda: self.delete_segrun(node))

            accepted = node.data(0, Qt.UserRole).accepted
            if accepted:
                menu.addAction('Make canonical',
                    lambda: self.make_canonical(node))

            menu.exec_(QCursor.pos())
        # could put other menu options for top-level nodes in an else here


    def set_fig_size(self, fig_w_inches, fig_h_inches) -> None:
        self.fig.set_size_inches(fig_w_inches, fig_h_inches)
        # TODO maybe some conversion factor so inches are actually inches on
        # scren (between mpl dots and pixels)
        # (assuming this is accurate)
        dpi = self.fig.dpi
        # TODO not sure why this isn't happening automatically, when figure
        # size changes. i feel like i'm missing something.
        # something like adjust size 
        self.mpl_canvas.resize(dpi * fig_w_inches, dpi * fig_h_inches)
        self.fig_w_inches = fig_w_inches
        self.fig_h_inches = fig_h_inches
        # So that the resize call is correct.
        self.current_zoom = 1.0


    # TODO TODO how to make zoom not affect size / placement of text relative to
    # other figure elments?? just calling self.mpl_canvas.resize
    # DOES change these things (zooming does not seem to apply to text)
    def zoom_canvas(self, delta) -> None:
        # TODO implement some lockout time to let slow drawing finish
        # TODO TODO maybe increment delta in eventFilter and then call
        # zoom_canvas after some delay from first event?
        # (so that a burst of scrolls is lumped together into one larger zoom)
        # TODO how to actually implement this...?
        if delta < 0:
            new_zoom = self.current_zoom - 0.2
            if new_zoom >= self.min_zoom:
                self.current_zoom = new_zoom
            else:
                return
        elif delta > 0:
            new_zoom = self.current_zoom + 0.2
            if new_zoom <= self.max_zoom:
                self.current_zoom = new_zoom
            else:
                return
        else:
            return
        dpi = self.fig.dpi
        self.mpl_canvas.resize(
            int(round(self.current_zoom * dpi * self.fig_w_inches)),
            int(round(self.current_zoom * dpi * self.fig_h_inches))
        )


    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel:
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ControlModifier:
                delta = event.angleDelta().y()
                self.zoom_canvas(delta)
                return True
        return super(Segmentation, self).eventFilter(source, event)


    def display_params_editable(self, editable) -> None:
        self.plot_corrs_btn.setEnabled(editable)
        self.plot_traces_btn.setEnabled(editable)


    def set_boolean(self, key, qt_value) -> None:
        # TODO share this if.. w/ cnmf_set_boolean somehow?
        if qt_value == 0:
            new_value = False
        elif qt_value == 2:
            new_value = True
        else:
            raise ValueError('unexpected checkbox signal output')
        # TODO more idiomatic way?
        setattr(self, key, new_value)

        
    def check_run_btn_enbl(self) -> None:
        self.params_changed = True
        if (not self.cnmf_running and self.movie is not None):
            self.run_cnmf_btn.setEnabled(True)


    # TODO after implementing per-type, see if can be condensed to one function
    # for all types
    def cnmf_set_boolean(self, group, key, qt_value) -> None:
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

        # TODO might want to actually check the value is different
        self.check_run_btn_enbl()


    # TODO so looks like this can be collapsed w/ list no problem? new name?
    # TODO wrap all these callbacks to enable/disable verbose stuff in one
    # place?
    def cnmf_set_from_spinbox(self, group, key, new_value) -> None:
        #print('Group:', group, 'Key:', key)
        #print('Old value:', self.params.get(group, key))
        self.params.set(group, {key: new_value})
        #print('New value:', self.params.get(group, key))
        # TODO might want to actually check the value is different
        self.check_run_btn_enbl()


    def cnmf_set_from_list(self, group, key, new_value) -> None:
        #print('Group:', group, 'Key:', key)
        #print('Old value:', self.params.get(group, key))
        self.params.set(group, {key: new_value})
        #print('New value:', self.params.get(group, key))
        # TODO might want to actually check the value is different
        self.check_run_btn_enbl()


    def cnmf_set_from_text(self, group, key, qt_line_edit) -> None:
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

            self.check_run_btn_enbl()


    def start_cnmf_worker(self) -> None:
        self.run_cnmf_btn.setEnabled(False)
        self.accept_cnmf_btn.setEnabled(False)
        self.reject_cnmf_btn.setEnabled(False)

        # Might consider moving this to end of cnmf call, so judgement can be
        # made while something else is running, but that might be kind of risky.

        # TODO notify user this is happening (and how?)? checkbox to not do
        # this?  and if checkbox is unticked, just store in db w/ accept as
        # null?
        '''
        if self.accepted is None and self.cnm is not None:
            self.reject_cnmf()
        '''

        # TODO separate button to cancel? change run-button to cancel?

        # TODO maybe don't do this (yet)?
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
        # TODO TODO implement. may require allowing callbacks to be passed into
        # cnmf code to report progress?
        ####worker.signals.progress.connect(self.progress_fn)

        self.parameter_json = self.params.to_json()
        # TODO make names of these more similar
        self.cnmf_start = time.time()
        self.run_at = datetime.fromtimestamp(self.cnmf_start)

        self.params_changed = False
        self.cnmf_running = True

        self.threadpool.start(worker)


    def run_cnmf(self) -> None:
        print('Running CNMF', flush=True)
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
            assert self.params_copy == self.params, \
                'CNMF changed params on init'

        # TODO what to do in case of memory error? 
        # is the cost / benefit there for trying to save intermediate results?
        # or just clear them all out?

        # TODO maybe use fit_file for certain ways of getting here in the gui?
        # TODO check dims / C/F order

        try:
            # From CNMF docs, about first arg to fit:
            # "images : mapped np.ndarray of shape (t,x,y[,z])"
            # (thought it doesn't actually need to be a memory mapped file)
            self.cnm.fit(self.movie,
                intermediate_footprints=self.plot_intermediates)

        # TODO test this case
        except MemoryError as err:
            traceback.print_exc()
            # TODO maybe log this / print traceback regardless
            # TODO get cnmf output not handling this appropriately. fix.
            self.cnm = None
            # TODO was this not actually printing? cnm was none...
            raise

        # TODO see which parameters are changed?
        if err_if_cnmf_changes_params:
            assert self.params_copy == self.params, 'CNMF changed params in fit'

        # TODO maybe have a widget that shows the text output from cnmf?


    # TODO TODO actually provide a way to only initialize cnmf, to test out
    # various initialization procedures (though only_init arg doesn't actually
    # seem to accomplish this correctly)
    def get_recording_dfs(self) -> None:
        """
        Sets:
        - self.df_over_f
        - self.start_frames
        - self.stop_frames
        - self.presentation_dfs (list # trials long)
        - self.comparison_dfs (list # trials long)
        - self.footprint_df
        """
        # TODO could maybe compute my own df/f from this if i'm worried...
        # frame number, cell -> value
        raw_f = self.cnm.estimates.C.T

        # TODO TODO TODO to copy what Remy's matlab script does, need to detrend
        # within each "block"
        if self.cnm.estimates.F_dff is None:
            # quantileMin=8, frames_window=500, flag_auto=True, use_fast=False,
            # (a, b, C, f, YrA)
            # TODO TODO TODO don't i want to use extract_... though, since more
            # exact?
            self.cnm.estimates.detrend_df_f()

        self.df_over_f = self.cnm.estimates.F_dff.T

        # TODO why 474 x 4 + 548 in one case? i thought frame numbers were
        # supposed to be more similar... (w/ np.diff(odor_onset_frames))
        first_onset_frame_offset = \
            self.odor_onset_frames[0] - self.block_first_frames[0]

        n_frames, n_cells = self.df_over_f.shape
        # would have to pass footprints back / read from sql / read # from sql
        ##assert n_cells == n_footprints

        start_frames = np.append(0,
            self.odor_onset_frames[1:] - first_onset_frame_offset)
        stop_frames = np.append(
            self.odor_onset_frames[1:] - first_onset_frame_offset - 1, n_frames)
        lens = [stop - start for start, stop in zip(start_frames, stop_frames)]

        assert self.frame_times.shape[0] == n_frames

        print(start_frames)
        print(stop_frames)
        # TODO find where the discrepancies are!
        print(sum(lens))
        print(n_frames)
        # TODO TODO TODO should i assert that all lens are the same????
        # (it seems analysis code requires this...)

        # TODO assert here that all frames add up / approx

        # TODO TODO either warn or err if len(start_frames) is !=
        # len(odor_pair_list)

        odor_id_pairs = [(o1,o2) for o1,o2 in
            zip(self.odor1_ids, self.odor2_ids)]
        print('odor_id_pairs:', odor_id_pairs)

        self.start_frames = start_frames
        self.stop_frames = stop_frames

        self.presentation_dfs = []
        self.comparison_dfs = []
        comparison_num = -1
        for i in range(len(start_frames)):
            if i % self.presentations_per_block == 0:
                comparison_num += 1
                repeat_nums = {id_pair: 0 for id_pair in odor_id_pairs}

            # TODO TODO also save to csv/flat binary/hdf5 per (date, fly,
            # thorimage) (probably at most, only when actually accepted.
            # that or explicit button for it.)
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
            assert len(presentation_frametimes) > 1

            odor_pair = odor_id_pairs[i]
            odor1, odor2 = odor_pair
            repeat_num = repeat_nums[odor_pair]
            repeat_nums[odor_pair] = repeat_num + 1

            offset_frame = self.odor_offset_frames[i]
            print('offset_frame:', offset_frame)
            assert offset_frame > direct_onset_frame

            # TODO check that all frames go somewhere and that frames aren't
            # given to two presentations. check they stay w/in block boundaries.
            # (they don't right now. fix!)

            # TODO share more of this w/ dataframe creation below, unless that
            # table is changed to just reference presentation table
            presentation = pd.DataFrame({
                # TODO fix hack
                'temp_presentation_id': [i],
                'prep_date': [self.date],
                'fly_num': self.fly_num,
                'recording_from': self.started_at,
                'analysis': self.run_at,
                'comparison': comparison_num,
                'odor1': odor1,
                'odor2': odor2,
                'repeat_num': repeat_num,
                'odor_onset_frame': direct_onset_frame,
                'odor_offset_frame': offset_frame,
                'from_onset': [[float(x) for x in presentation_frametimes]]
            })

            # TODO TODO TODO assert that len(presentation_frametimes)
            # == stop_frame - start_frame (off-by-one?)
            # TODO (it would fail now) fix!!
            # maybe this is a failure to merge correctly later???
            # b/c presentation frametimes seems to be defined to be same length
            # above... same indices...
            # (unless maybe self.frame_times is sometimes shorter than
            # self.df_over_f, etc)

            '''
            presentation_dff = self.df_over_f[start_frame:stop_frame, :]
            presentation_raw_f = raw_f[start_frame:stop_frame, :]
            '''

            # TODO TODO TODO fix / delete hack!!
            # TODO probably just need to more correctly calculate stop_frame?
            # (or could also try expanding frametimes to include that...)
            actual_frametimes_slice_len = len(presentation_frametimes)
            # TODO TODO why didn't this fix it!?!?!? (did it?)
            stop_frame = start_frame + actual_frametimes_slice_len
            presentation_dff = self.df_over_f[start_frame:stop_frame, :]
            presentation_raw_f = raw_f[start_frame:stop_frame, :]

            # TODO TODO TODO if these all start off as the same length,
            # what ultimately compresses the frame times to len 680?
            # just do the same thing w/ the other two?
            # or don't do it w/ frame times?
            print(presentation_frametimes.shape)
            print(presentation_raw_f.shape)
            print(presentation_dff.shape)

            ############import ipdb; ipdb.set_trace()
            #

            # Assumes that cells are indexed same here as in footprints.
            cell_dfs = []
            for cell_num in range(n_cells):

                cell_dff = presentation_dff[:, cell_num].astype('float32')
                cell_raw_f = presentation_raw_f[:, cell_num].astype('float32')

                cell_dfs.append(pd.DataFrame({
                    # TODO maybe rename / do in a less hacky way
                    'temp_presentation_id': [i],
                    ###'presentation_id': [presentation_id],
                    'recording_from': [self.started_at],
                    'segmentation_run': [self.run_at],
                    'cell': [cell_num],
                    'df_over_f': [[float(x) for x in cell_dff]],
                    'raw_f': [[float(x) for x in cell_raw_f]]
                }))
            response_df = pd.concat(cell_dfs, ignore_index=True)

            # TODO maybe draw correlations from each of these, as i go?
            # (would still need to do block by block, not per trial)

            self.presentation_dfs.append(presentation)
            # TODO rename...
            self.comparison_dfs.append(response_df)

            print('Done processing presentation {}'.format(i))

        # TODO TODO do all footprints stuff w/o converting to full array!
        # x,y,n_footprints
        footprints = self.cnm.estimates.A.toarray()

        # Assuming equal number along both dimensions.
        pixels_per_side = int(np.sqrt(footprints.shape[0]))
        n_footprints = footprints.shape[1]

        footprints = np.reshape(footprints,
            (pixels_per_side, pixels_per_side, n_footprints))
        
        footprint_dfs = []
        for cell_num in range(n_footprints):
            sparse = coo_matrix(footprints[:,:,cell_num])
            footprint_dfs.append(pd.DataFrame({
                'recording_from': [self.started_at],
                'segmentation_run': [self.run_at],
                'cell': [cell_num],
                # Can be converted from lists of Python types, but apparently
                # not from numpy arrays or lists of numpy scalar types.
                # TODO check this doesn't transpose things
                # TODO just move appropriate casting to my to_sql function,
                # and allow having numpy arrays (get type info from combination
                # of that and the database, like in other cases)
                'x_coords': [[int(x) for x in sparse.col.astype('int16')]],
                'y_coords': [[int(x) for x in sparse.row.astype('int16')]],
                'weights': [[float(x) for x in sparse.data.astype('float32')]]
            }))
        self.footprint_df = pd.concat(footprint_dfs, ignore_index=True)


    # TODO some kind of test option / w/ test data for this part that doesn't
    # require actually running cnmf
    # TODO rename to "process_..." or something?
    # TODO make this fn not block gui? did cnmf_done not block gui? maybe that's
    # a reason to keep it? or put most of this stuff in the worker?
    def get_cnmf_output(self) -> None:
        # TODO TODO TODO w/ python version of cnmf, as i'm using it, do i need
        # to explicitly order components to get same ordering as in matlab ver?

        self.run_len_seconds = time.time() - self.cnmf_start
        print('CNMF took {:.1f}s'.format(self.run_len_seconds))
        # TODO maybe time this too? (probably don't store in db tho)
        # at temporarily, to see which parts are taking so long...
        print('Processing the output...')

        # TODO TODO allow toggling between type of background image shown
        # (radio / combobox for avg, snr, etc? "local correlations"?)
        # TODO TODO use histogram equalized avg image as one option
        img = self.avg

        only_init = self.params_copy.get('patch', 'only_init')

        n_footprint_axes = 4 if self.plot_intermediates and not only_init else 1

        self.display_params_editable(False)

        # TODO checkbox for plot_intermediates
        plot_correlations = self.plot_correlations
        plot_traces = self.plot_traces

        w_inches_footprint_axes = 3
        h_inches_per_footprint_ax = 3
        w_inches_per_corr = 3
        h_inches_corrs = 2 * w_inches_per_corr
        w_inches_per_traceplot = 8
        h_inches_traceplots = 10
        
        w_inches_corr = w_inches_per_corr * self.n_blocks
        h_inches_footprint_axes = h_inches_per_footprint_ax * n_footprint_axes
        w_inches_traceplots = w_inches_per_traceplot * self.n_blocks
        # TODO could set this based on whether i want 1 / both orders
        #h_inches_per_traceplot = 5
        #h_inches_traceplots = h_inches_per_traceplot * 

        widths = [w_inches_footprint_axes]
        heights = [h_inches_footprint_axes]
        if plot_correlations:
            widths.append(w_inches_corr)
            heights.append(h_inches_corrs)

        if plot_traces:
            widths.append(w_inches_traceplots)
            heights.append(h_inches_traceplots)

        fig_w_inches = max(widths)
        fig_h_inches = sum(heights)

        self.set_fig_size(fig_w_inches, fig_h_inches)
        # TODO maybe there isn't always too much space between this and first
        # thing, but there is in some cases. avoid if possible.
        # TODO TODO fix suptitle so it plays nicely w/ tight_layout
        # or do some other way. right now, in the 1 footprint ax case,
        # w/ no corrs or traces, suptitle is right on top of plot title!
        self.fig.suptitle(self.recording_title)

        footprint_rows = 2
        if plot_correlations:
            corr_rows = 2
        else:
            corr_rows = 0
        if plot_traces:
            top_components = True
            # will probably be more meaningful once i can restrict to responders
            random_components = False
            trace_rows = 2 * sum([top_components, random_components])
        else:
            trace_rows = 0

        # TODO set ratio of footprint:corr rows based on n_footprint_axes
        # (& corr types)

        gs_rows = sum([footprint_rows, corr_rows, trace_rows])
        gs = self.fig.add_gridspec(gs_rows, 1, hspace=0.4, wspace=0.05)

        # TODO maybe redo gs... maybe it should be derived from the
        # h/w_inches stuff? (if everything gets 2/6 why not just 1/3...?)
        footprint_slice = gs[:2, :]

        footprint_gs = footprint_slice.subgridspec(
            n_footprint_axes, 1, hspace=0, wspace=0)

        axs = []
        ax0 = None
        for i in range(footprint_gs._nrows):
            if ax0 is None:
                ax = self.fig.add_subplot(footprint_gs[i])
            else:
                ax = self.fig.add_subplot(footprint_gs[i],
                    sharex=ax0, sharey=ax0)

            axs.append(ax)
        contour_axes = np.array(axs)

        if plot_correlations:
            # (end slice is not included, as always, so it's same size as above)
            corr_slice = gs[2:4, :]
            # 2 rows: one for correlation matrices ordered as in experiment,
            # and the other for matrices ordered by odor
            corr_gs = corr_slice.subgridspec(2, self.n_blocks,
                hspace=0.4, wspace=0.1)

            axs = []
            for i in range(corr_gs._nrows):
                axs.append([])
                for j in range(corr_gs._ncols):
                    # TODO maybe still do this? anyway way to indicate the
                    # matrix intensity scale should be shared (but that's not x
                    # or y, right?)?
                    '''
                    if ax0 is None:
                        ax = fig.add_subplot(corr_gs[i])
                    else:
                        ax = fig.add_subplot(corr_gs[i], sharex=ax0, sharey=ax0)
                    '''
                    ax = self.fig.add_subplot(corr_gs[i,j])
                    axs[-1].append(ax)
            corr_axes = np.array(axs)

        if plot_traces:
            # TODO maybe stack each block vertically here, and then make total
            # rows in gs depend on # of blocks??? (maybe put corrs to the side?)
            # TODO might make sense to have one grid unit per set of two
            # subplots, so space for shared title above the two is separate from
            # space between the two (e.g. the "Top components" thing)
            all_blocks_trace_gs = gs[4:, :].subgridspec(trace_rows,
                self.n_blocks, hspace=0.3, wspace=0.15)

        for i in range(n_footprint_axes):
            contour_ax = contour_axes[i]
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
            if i == n_footprint_axes - 1 and not only_init:
                contour_ax.set_title('Final estimate')
                A = self.cnm.estimates.A

            elif not self.plot_intermediates and i == 0 and only_init:
                contour_ax.set_title('Initialization')
                A = self.cnm.estimates.A

            caiman.utils.visualization.plot_contours(A, img, ax=contour_ax,
                display_numbers=False, colors='r', linewidth=1.0)

            self.mpl_canvas.draw()

        ###################################################################
        # TODO defer this as much as possible
        # (move later, don't compute traces if not needed for plots / accept)
        self.get_recording_dfs()

        if plot_correlations or plot_traces:

            # TODO TODO TODO make this configurable in gui / have correlations
            # update (maybe not alongside CNMF parameters, to avoid confusion?)
            response_calling_s = 3.0

            # TODO maybe make this abbreviation making a fn
            # TODO maybe use abbreviation that won't need a separate table to be
            # meaningful...
            # TODO sort s.t. always goes A,B,C in odor corr?
            presentations_df = pd.concat(self.presentation_dfs,
                ignore_index=True)
            # TODO check again that this also works in case where odors from
            # this experiment are new (weren't in db before)
            # (and maybe support some local analysis anyway that doesn't require
            # rt through the db...)
            presentations_df = u.merge_odors(presentations_df,
                self.db_odors.reset_index())
            # TODO maybe adapt to case where name2 might have only occurence of 
            # an odor, or name1 might be paraffin.
            # TODO TODO check this is actually in the order i want across blocks
            # (idk if name1,name2 are sorted / re-ordered somewhere)
            name1_unique = presentations_df.name1.unique()
            name2_unique = presentations_df.name2.unique()
            assert set(name2_unique) - set(name1_unique) == {'paraffin'}
            odor2abbrev = {o: chr(ord('A') + i)
                for i, o in enumerate(name1_unique)}
            # So that code detecting which combinations of name1+name2 are
            # monomolecular does not need to change.
            odor2abbrev['paraffin'] = 'paraffin'

            block_iter = list(range(self.n_blocks))
        else:
            block_iter = []

        for i in block_iter:
            # TODO maybe concat and only set whole df as instance variable in
            # get_recording_df? then use just as in kc_analysis all throughout
            # here?
            presentation_dfs = self.presentation_dfs[
                (self.presentations_per_block * i):
                (self.presentations_per_block * (i + 1))
            ]
            presentation_df = pd.concat(presentation_dfs,
                ignore_index=True)

            comparison_dfs = self.comparison_dfs[
                (self.presentations_per_block * i):
                (self.presentations_per_block * (i + 1))
            ]
            comparison_df = pd.concat(comparison_dfs,
                ignore_index=True)

            # TODO don't have separate instance variables for presentation_dfs
            # and comparison_dfs if i'm always going to merge here.
            # just merge before and then put in one instance variable.
            # (probably just keep name comparison_dfs)
            presentation_df['from_onset'] = presentation_df['from_onset'].apply(
                lambda x: np.array(x))

            presentation_df = u.merge_odors(presentation_df,
                self.db_odors.reset_index())
            presentation_df['name1'] = presentation_df.name1.map(odor2abbrev)
            presentation_df['name2'] = presentation_df.name2.map(odor2abbrev)

            presentation_df = u.merge_recordings(
                presentation_df, self.recordings)

            array_cols = ['raw_f', 'df_over_f']
            for ac in array_cols:
                comparison_df[ac] = comparison_df[ac].apply(
                    lambda x: np.array(x))
            array_cols = array_cols + ['from_onset']

            # TODO TODO why are lengths of raw_f and df_over_f not always equal
            # to the length of from_onset????? fix!!!!!
            # (they do seem to always be equal to themselves though)

            # TODO why does response_df (comparison_df) have recording_from col
            # again? (since kinda redundant w/ presentation...)
            comparison_df_shape_before = comparison_df.shape
            #print(comparison_df_shape_before)
            comparison_df = comparison_df.merge(presentation_df, how='left',
                left_on='temp_presentation_id', right_on='temp_presentation_id')
            comparison_df = comparison_df.drop(columns='temp_presentation_id')
            '''
            for c in presentation_df.columns:
                comparison_df[c] = presentation_df[c]
            '''
            #print(comparison_df.shape)
            # TODO del presentation_df?

            non_array_cols = comparison_df.columns.difference(array_cols)
            cell_response_dfs = []
            for _, cell_df in comparison_df.groupby(u.trial_cols + ['cell']):
                lens = cell_df[array_cols].apply(lambda x: x.str.len(),
                    axis='columns')
                # TODO delete try except
                try:
                    # TODO TODO what do i need to fix this????
                    # TODO was this originally supposed to be a check on
                    # uniqueness? (it's not now, whether or not that was orig
                    # intention)
                    assert len(lens) == 1
                except AssertionError:
                    import ipdb; ipdb.set_trace()

                length = lens.iat[0,0]
                assert (lens == length).all().all()

                # TODO maybe also do explosion stuff with index if the input
                # here has a meaningful index
                exploded = pd.DataFrame({c: np.repeat(cell_df[c].values, length)
                    for c in non_array_cols})

                for ac in array_cols:
                    exploded[ac] = np.concatenate(cell_df[ac].values)

                exploded.set_index(u.trial_cols, inplace=True)
                cell_response_dfs.append(exploded)

                # TODO maybe plot a set of traces here, as a sanity check?

            comparison_df = pd.concat(cell_response_dfs)
            comparison_df.reset_index(inplace=True) 

            frame2order = {f: o for o,f in
                enumerate(sorted(comparison_df.odor_onset_frame.unique()))}

            # TODO TODO exclude stuff that wasn't randomized w/in each
            # comparison?  (or at least be aware which are which...)

            comparison_df['order'] = \
                comparison_df.odor_onset_frame.map(frame2order)

            # TODO TODO add column mapping odors to order -> sort (index) on
            # that column + repeat_num to order w/ mixture last

            ###################################################################
            if plot_traces:
                n = 20

                # TODO or maybe just set show_footprints to false?
                footprints = u.merge_recordings(self.footprint_df,
                    self.recordings)
                footprints = u.merge_gsheet(footprints, df)
                footprints.set_index(u.recording_cols + ['cell'], inplace=True)

                # TODO TODO factor out response calling and also do that here,
                # so that random subset can be selected from responders, as in
                # kc_analysis? (maybe even put it in plot_traces?)
                # TODO maybe adapt whole mpl gui w/ updating db response calls
                # into here?

                # TODO TODO TODO + ensure components are actually ordered for
                # this approach (maybe scale is just innapropriate? radiobutton
                # to change it?)
                if top_components:
                    odor_order_trace_gs = all_blocks_trace_gs[0, i]

                    # TODO maybe allow passing movie in to not have to load it
                    # multiple times when plotting traces on same data?
                    # (then just use self.movie)
                    u.plot_traces(comparison_df, footprints=footprints,
                        gridspec=odor_order_trace_gs, n=n,
                        title='Top components')

                    presentation_order_trace_gs = all_blocks_trace_gs[1, i]
                    u.plot_traces(comparison_df, footprints=footprints,
                        gridspec=presentation_order_trace_gs,
                        order_by='presentation_order', n=n)

                if random_components:
                    if top_components:
                        orow = 2
                        prow = 3
                    else:
                        orow = 0
                        prow = 1

                    odor_order_trace_gs = all_blocks_trace_gs[orow, i]
                    u.plot_traces(comparison_df, footprints=footprints,
                        gridspec=odor_order_trace_gs, n=n, random=True,
                        title='Random components')

                    presentation_order_trace_gs = all_blocks_trace_gs[prow, i]
                    u.plot_traces(comparison_df, footprints=footprints,
                        gridspec=presentation_order_trace_gs,
                        order_by='presentation_order', n=n, random=True)

            ###################################################################
            if plot_correlations:
                # TODO TODO might want to only compute responders/criteria one
                # place, to avoid inconsistencies (so either move this section
                # into next loop and aggregate, or index into this stuff from
                # within that loop?)
                in_response_window = ((comparison_df.from_onset > 0.0) &
                    (comparison_df.from_onset <= response_calling_s))

                # TODO TODO include from_onset col then compute mean?
                window_df = comparison_df.loc[in_response_window,
                    cell_cols + ['order','from_onset','df_over_f']]

                # TODO maybe move this to bottom, around example trace plotting
                window_by_trial = \
                    window_df.groupby(cell_cols + ['order'])['df_over_f']

                window_trial_means = window_by_trial.mean()
                # TODO rename to 'mean_df_over_f' or something, to avoid
                # confusion
                trial_by_cell_means = window_trial_means.to_frame().pivot_table(
                    index=['name1','name2','repeat_num','order'],
                    columns='cell', values='df_over_f').T

                trial_mean_presentation_order = \
                    trial_by_cell_means.sort_index(axis=1, level='order')

                odor_order_trial_mean_corrs = trial_by_cell_means.corr()
                presentation_order_trial_mean_corrs = \
                    trial_mean_presentation_order.corr()


                corr_cbar_label = (r'Mean response $\frac{\Delta F}{F}$' +
                        ' correlation')
                presentation_order_ax = corr_axes[0, i]

                ticklabels = u.matlabels(presentation_order_trial_mean_corrs,
                    u.format_mixture)

                # TODO TODO use titles to say which two odors it was
                u.matshow(presentation_order_trial_mean_corrs,
                    ticklabels=ticklabels,
                    colorbar_label=corr_cbar_label,
                    #title=fly_comparison_title,
                    ax=presentation_order_ax,
                    fontsize=6)
                self.mpl_canvas.draw()


                odor_order_ax = corr_axes[1, i]

                ticklabels = u.matlabels(odor_order_trial_mean_corrs,
                    u.format_mixture)

                u.matshow(odor_order_trial_mean_corrs,
                    ticklabels=ticklabels,
                    group_ticklabels=True,
                    colorbar_label=corr_cbar_label,
                    #title=fly_comparison_title,
                    ax=odor_order_ax,
                    fontsize=6)
                self.mpl_canvas.draw()

        ###################################################################
        if plot_correlations or plot_traces:
            abbrev2odor = {v: k for k, v in odor2abbrev.items()}
            print('\nOdor abbreviations:')
            for k in sorted(abbrev2odor.keys()):
                if k != 'paraffin':
                    print('{}: {}'.format(k, abbrev2odor[k]))
            print('')

        # TODO maybe delete this...
        self.fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
        self.mpl_canvas.draw()
        # TODO maybe allow toggling same pane between avg and movie?
        # or separate pane for movie?
        # TODO use some non-movie version of pyqtgraph ImageView for avg,
        # to get intensity sliders? or other widget for that?
        self.accepted = None
        self.cnmf_running = False

        self.display_params_editable(True)

        if self.cnm is not None:
            self.accept_cnmf_btn.setEnabled(True)
            self.reject_cnmf_btn.setEnabled(True)

            if self.params_changed:
                self.run_cnmf_btn.setEnabled(True)


    # TODO maybe refactor this and save_default_params a little...
    # if i want to support both json_str and cnmfparams, might make sense to
    # only have that logic one place
    def save_json(self, json_str, *args) -> None:
        """Saves a str to a filename.

        If output_filename is not provided, a dialog will pop up asking user
        where to save the file.
        """
        if len(args) == 0:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            output_filename, _ = QFileDialog.getSaveFileName(self, 'Save to...',
                '', 'JSON (*.json)', options=options)
        elif len(args) == 1:
            output_filename = args[0]
        else:
            raise ValueError('incorrect number of arguments')

        # TODO maybe validate json_str first?
        # TODO maybe nicely format it in the file, so it's more human readable?
        with open(output_filename, 'w') as f:
            f.write(json_str)


    def save_default_params(self, *args) -> None:
        """Saves CNMF params to file in JSON format.

        If optional arg is passed, it is assumed to be a string of valid CNMF
        param JSON and it is saved as-is. Otherwise, CNMFParams.to_json is used
        on self.params
        """
        print('Writing new default parameters to {}'.format(
            self.default_json_params))
        # TODO TODO test round trip before terminating w/ success
        if len(args) == 0:
            self.params.to_json(self.default_json_params)
        elif len(args) == 1:
            json_str = args[0]
            self.save_json(json_str, self.default_json_params)
        else:
            raise ValueError('incorrect number of arguments')


    # TODO TODO add support for deleting presentations from db if reject
    # something that was just accepted?


    # TODO move core of this to util and just wrap it here
    def upload_segmentation_info(self, accepted: bool) -> None:
        # TODO maybe visually indicate which has been selected already?
        # TODO TODO check whether row in tree view has been created
        # / other state indicating whether already in gui
        # if so, just use update_seg_accepted fn
        run_info = {
            'run_at': [self.run_at],
            'recording_from': self.started_at,
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
        run_info['accepted'] = accepted
        run = pd.DataFrame(run_info)
        run.set_index('run_at', inplace=True)

        # TODO depending on what table is in method callable, may need to make
        # pd index match sql pk?
        # TODO test that result is same w/ or w/o method in case where row did
        # not exist, and that read shows insert worked in w/ method case
        if self.ACTUALLY_UPLOAD:
            run.to_sql('analysis_runs', u.conn, if_exists='append',
                method=u.pg_upsert)

        # TODO worth preventing (attempts to) insert code versions and
        # pairings with analysis_runs, or is that premature optimization?
        if self.accepted is not None:
            # The remainder of the metadata must have already been uploaded,
            # when self.accepted was first set.
            return

        # TODO nr_code_versions, rig_code_versions (both lists of dicts)
        # TODO ti_code_version (dict)
        if self.tiff_fname.endswith('_nr.tif'):
            mocorr_version_varname = 'nr_code_versions'
        elif self.tiff_fname.endswith('_rig.tif'):
            mocorr_version_varname = 'rig_code_versions'
        else:
            raise NotImplementedError

        ti_code_version = u.get_matfile_var(self.matfile, 'ti_code_version')
        mocorr_code_versions = u.get_matfile_var(self.matfile,
            mocorr_version_varname, require=False)

        code_versions = (common_code_versions + ti_code_version +
            mocorr_code_versions)

        # TODO maybe impute missing mocorr version in some cases?

        if self.ACTUALLY_UPLOAD:
            u.upload_analysis_info(self.run_at, code_versions)

        png_buff = BytesIO()
        self.fig.savefig(png_buff, format='png')
        svg_buff = BytesIO()
        self.fig.savefig(svg_buff, format='svg')

        # I haven't yet figured out how to deserialize these in regular
        # interactive pyplot, but one test case did work in the same kind of Qt
        # setting (which is main goal anyway).
        fig_buff = BytesIO()
        pickle.dump(self.fig, fig_buff)

        # TODO delete. for debugging.
        print('png_buff nbytes:', png_buff.getbuffer().nbytes)
        print('svg_buff nbytes:', svg_buff.getbuffer().nbytes)
        print('fig_buff nbytes:', fig_buff.getbuffer().nbytes)
        print('png_buff sizeof:', sys.getsizeof(png_buff))
        print('svg_buff sizeof:', sys.getsizeof(svg_buff))
        print('fig_buff sizeof:', sys.getsizeof(fig_buff))
        #

        # TODO TODO failed a few times on this insert, possibly b/c fig size
        # try inserting one at a time? fallback to saving to nas and including
        # filepath in another column?

        segmentation_run = pd.DataFrame({
            'run_at': [self.run_at],
            'output_fig_png': png_buff.getvalue(),
            'output_fig_svg': svg_buff.getvalue(),
            'output_fig_mpl': fig_buff.getvalue(),
            'run_len_seconds': self.run_len_seconds
        })
        segmentation_run.set_index('run_at', inplace=True)

        # TODO are the reset_index calls necessary?
        segrun_row = segmentation_run.reset_index().merge(
            run.reset_index()).iloc[0]
        self.current_item = \
            self.add_segrun_widget(self.current_item, segrun_row)

        if self.ACTUALLY_UPLOAD:
            segmentation_run.to_sql('segmentation_runs', u.conn,
                if_exists='append', method=u.pg_upsert)

        # TODO filter out footprints less than a certain # of pixels in cnmf?
        # (is 3 pixels really reasonable?)
        if self.ACTUALLY_UPLOAD:
            u.to_sql_with_duplicates(self.footprint_df, 'cells')


    def update_seg_color(self, segrun_treeitem, accepted) -> None:
        if accepted:
            color = self.accepted_color
        else:
            color = self.rejected_color
        segrun_treeitem.setBackground(0, QColor(color))

        # This includes the only case where parents color needs to change
        # TO accepted.
        parent = segrun_treeitem.parent()
        if accepted:
            parent.setBackground(0, QColor(color))
        else:
            any_accepted = False
            for i in range(parent.childCount()):
                if parent.child(i).data(0, Qt.UserRole).accepted:
                    any_accepted = True
                    break
            if not any_accepted:
                parent.setBackground(0, QColor(color))


    def update_seg_accepted(self, segrun_treeitem, accepted) -> None:
        row = segrun_treeitem.data(0, Qt.UserRole)
        row.accepted = accepted

        # TODO maybe this should be row.run_at?
        sql = ("UPDATE analysis_runs SET accepted = " +
            "{} WHERE run_at = '{}'").format(accepted,
            self.run_at)
        ret = u.conn.execute(sql)

        self.update_seg_color(segrun_treeitem, accepted)

        if accepted:
            # Starts out this way
            canonical = False
        else:
            canonical = None
        self.mark_canonical(segrun_treeitem, canonical)


    # TODO maybe make all accept / reject buttons gray until current accept /
    # reject finishes running (mostly to avoid having to think about whether not
    # doing so could possibly cause a problem)?
    def accept_cnmf(self) -> None:
        if self.accepted:
            return

        if self.accepted is not None:
            self.update_seg_accepted(self.current_item, True)
            self.accepted = True
            return

        self.accept_cnmf_btn.setEnabled(False)
        self.reject_cnmf_btn.setEnabled(False)

        self.upload_segmentation_info(True)

        # TODO just calculate metadata outright here?
            
        # TODO TODO save file to nas (particularly so that it can still be there
        # if database gets wiped out...) (should thus include parameters
        # [+version?] info)

        # TODO and refresh stuff in validation window s.t. this experiment now
        # shows up

        # TODO maybe also allow more direct passing of this data to other tab

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

        # TODO delete me
        print('saving CNMF state to cnmf_state.p for debugging', flush=True)
        # intended to use this to find best detrend / extract dff method
        try:
            state = {
                'Yr': Yr,
                'A': ests.A,
                'C': ests.C,
                'bl': ests.bl,
                'b': ests.b,
                'f': ests.f,
                'df_over_f': self.df_over_f,
                # TODO raw_f too? or already included in one of these things?
                'start_frames': self.start_frames,
                'stop_frames': self.stop_frames,
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
        print('done saving cnmf_state.p', flush=True)
        #

        # TODO TODO TODO why did i need this separate from checking
        # self.accepted is None???
        if self.uploaded_presentations:
            # Just to skip to end of function.
            presentation_dfs = []
            comparison_dfs = []
        else:
            presentation_dfs = self.presentation_dfs
            comparison_dfs = self.comparison_dfs

        # TODO could also add some check / cleanup routines for orphaned
        # rows in presentations table (and maybe some other tables)
        # maybe share w/ code that checks distinct to decide whether to
        # load / analyze?
        key_cols = [
            'prep_date',
            'fly_num',
            'recording_from',
            'analysis',
            'comparison',
            'odor1',
            'odor2',
            'repeat_num'
        ]

        if self.ACTUALLY_UPLOAD:
            for presentation_df, comparison_df in zip(
                presentation_dfs, comparison_dfs):

                u.to_sql_with_duplicates(presentation_df.drop(
                    columns='temp_presentation_id'), 'presentations')

                # TODO how exactly is this supposed to work in not
                # self.ACTUALLY_UPLOAD case anyway? it wouldn't necessarily,
                # right?
                db_presentations = pd.read_sql('presentations', u.conn,
                    columns=(key_cols + ['presentation_id']))

                presentation_ids = (db_presentations[key_cols] ==
                    presentation_df[key_cols].iloc[0]).all(axis=1)

                assert presentation_ids.sum() == 1, \
                    'presentation_id could not be determined uniquely'

                presentation_id = db_presentations.loc[presentation_ids,
                    'presentation_id'].iat[0]

                comparison_df['presentation_id'] = presentation_id

                u.to_sql_with_duplicates(comparison_df.drop(
                    columns='temp_presentation_id'), 'responses')

        self.uploaded_presentations = True

        # TODO TODO mark as canonical if first accepted / most recently
        # accepted?
        # (maybe unless there's one explicitly marked as canonical?)

        self.accept_cnmf_btn.setEnabled(True)
        self.reject_cnmf_btn.setEnabled(True)

        self.current_item.parent().setBackground(0, QColor(self.accepted_color))
        self.accepted = True
        print('accepted')


    def reject_cnmf(self):
        if self.accepted == False:
            return

        if self.accepted is not None:
            self.update_seg_accepted(self.current_item, False)
            self.accepted = False
            return

        self.accept_cnmf_btn.setEnabled(False)
        self.reject_cnmf_btn.setEnabled(False)

        self.upload_segmentation_info(False)

        self.accept_cnmf_btn.setEnabled(True)
        self.reject_cnmf_btn.setEnabled(True)

        self.accepted = False
        print('rejected')


    # TODO maybe support save / loading cnmf state w/ their save/load fns w/
    # buttons in the gui? (maybe to some invisible cache?)
    # (would need to fix cnmf save (and maybe load too) fn(s))


    def handle_treeitem_click(self):
        curr_item = self.sender().currentItem()
        self.current_item = curr_item
        if curr_item.parent() is None:
            self.open_recording(curr_item)
        # TODO check this works for non-toplevel nodes
        else:
            self.open_segmentation_run(curr_item)


    def delete_other_param_widgets(self) -> None:
        if self.param_display_widget is None:
            return

        self.param_widget_stack.setCurrentIndex(0)
        self.param_widget_stack.removeWidget(self.param_display_widget)

        # maybe remove this sip bit... not sure it's necessary
        sip.delete(self.param_display_widget)
        # alternative would probably be:
        # self.param_display_widget.deleteLater()
        # see: stackoverflow.com/questions/5899826

        self.param_display_widget = None


    # TODO TODO test case where this or open_recording are triggered
    # when cnmf is running / postprocessing
    # (what happens? what should happen? maybe just disable callbacks during?)
    def open_segmentation_run(self, item):
        row = item.data(0, Qt.UserRole)
        self.run_at = row.run_at
        self.accepted = row.accepted

        self.delete_other_param_widgets()

        # TODO possible to implement this w/o interring w/ other state?
        # (so movie can stay loaded, and only change when selecting another
        # experiment, etc)
        # (now i'm clearing stuff just to be safe)
        self.cnm = None
        self.movie = None

        # TODO delete any existing widgets in stack beyond first
        self.param_display_widget = self.make_cnmf_param_widget(row.parameters,
            editable=False)

        self.param_widget_stack.addWidget(self.param_display_widget)
        self.param_widget_stack.setCurrentIndex(1)

        # TODO also load correct data params
        # is it a given that param json reflects correct data params????
        # if not, may need to model after open_recording

        # maybe this is not worth it / necessary
        self.fig.clear()

        self.fig = pickle.load(BytesIO(row.output_fig_mpl))
        self.mpl_canvas.figure = self.fig
        self.fig.canvas = self.mpl_canvas
        fig_w_inches, fig_h_inches = self.fig.get_size_inches()
        self.set_fig_size(fig_w_inches, fig_h_inches)
        # TODO test case where canvas was just drawing something larger
        # (is draw area not fully updated?)
        # TODO need tight_layout?
        self.mpl_canvas.draw()

        # TODO TODO should probably delete any traces in db if select reject
        # (or maybe just ignore and let other scripts clean orphaned stuff up
        # later? so that we don't have to regenerate traces if we change our
        # mind again and want to label as accepted...)
        self.accept_cnmf_btn.setEnabled(True)
        self.reject_cnmf_btn.setEnabled(True)


    # TODO maybe selecting two (/ multiple) analysis runs then right clicking
    # (or other way to invoke action?) and diffing parameters
    # TODO maybe also allow showing footprints overlayed or comparing somehow
    # (in same viewer?)


    def open_recording(self, item):
        # TODO maybe use setData and data instead?
        idx = self.data_tree.indexOfTopLevelItem(item)
        tiff = self.motion_corrected_tifs[idx]

        self.delete_other_param_widgets()

        start = time.time()

        tiff_dir, tiff_just_fname = split(tiff)
        analysis_dir = split(tiff_dir)[0]
        full_date_dir, fly_dir = split(analysis_dir)
        date_dir = split(full_date_dir)[-1]

        date = datetime.strptime(date_dir, '%Y-%m-%d')
        fly_num = int(fly_dir)
        thorimage_id = '_'.join(tiff_just_fname.split('_')[:-1])

        self.recording_title = '{}/{}/{}'.format(
            date_dir, fly_num, thorimage_id)
        print('')
        print(self.recording_title)

        # Trying all the operations that need to find files before setting any
        # instance variables, so that if those fail, we can stay on the current
        # data if we want (without having to reload it).
        ########################################################################
        # Start stuff more likely to fail (missing file, etc)
        ########################################################################

        mat = join(analysis_dir, rel_to_cnmf_mat, thorimage_id + '_cnmf.mat')

        try:
            ti = u.load_mat_timing_information(mat)
        except matlab.engine.MatlabExecutionError as e:
            # TODO inspect error somehow to see if it's a memory error?
            # -> continue if so
            # TODO print to stderr
            print(e)
            return

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

        data_params = u.cnmf_metadata_from_thor(tiff)

        started_at = \
            datetime.fromtimestamp(float(xml_root.find('Date').attrib['uTime']))

        # TODO see part of populate_db.py in this section to see how data
        # explorer list elements might be colored to indicate they have already
        # been run? or just get everything w/ pandas and color all from there?
        # TODO pane to show previous analysis runs of currently selected
        # experiment, or not worth it since maybe only want one accepted per?

        # TODO need to search db for (at least) canonical_segmentation, if i
        # want to set state of all this as data
        # TODO TODO could ignore case where recording wasn't already in db tho
        # TODO TODO upload full_frame_avg_trace like in populate_db
        self.recordings = pd.DataFrame({
            'started_at': [started_at],
            'thorsync_path': [thorsync_dir],
            'thorimage_path': [thorimage_dir],
            'stimulus_data_path': [stimulus_data_path]#,
            #'full_frame_avg_trace': 
            #'canonical_segmentation': None
        })
        # TODO at least put behind self.ACTUALLY_UPLOAD?
        # TODO maybe defer this to accepting?
        # TODO rename to singular?
        u.to_sql_with_duplicates(self.recordings, 'recordings')

        n_repeats = int(data['n_repeats'])

        # The 3 is because 3 odors are compared in each repeat for the
        # natural_odors project.
        presentations_per_repeat = 3

        presentations_per_block = n_repeats * presentations_per_repeat

        if pd.isnull(recording['first_block']):
            first_block = 0
        else:
            first_block = int(recording['first_block']) - 1

        if pd.isnull(recording['last_block']):
            n_full_panel_blocks = \
                int(len(data['odor_pair_list']) / presentations_per_block)

            last_block = n_full_panel_blocks - 1

        else:
            last_block = int(recording['last_block']) - 1

        # TODO maybe use subset here too, to be consistent w/ which mixtures get
        # entered...
        odors = pd.DataFrame({
            'name': data['odors'],
            'log10_conc_vv': [0 if x == 'paraffin' else
                natural_odors_concentrations.at[x,
                'log10_vial_volume_fraction'] for x in data['odors']]
        })
        u.to_sql_with_duplicates(odors, 'odors')

        # TODO make unique id before insertion? some way that wouldn't require
        # the IDs, but would create similar tables?

        self.db_odors = pd.read_sql('odors', u.conn)
        # TODO TODO in general, the name alone won't be unique, so use another
        # strategy
        self.db_odors.set_index('name', inplace=True)

        first_presentation = first_block * presentations_per_block
        last_presentation = (last_block + 1) * presentations_per_block - 1

        odor_pair_list = \
            data['odor_pair_list'][first_presentation:(last_presentation + 1)]

        assert (len(odor_pair_list) %
            (presentations_per_repeat * n_repeats) == 0)

        # TODO invert to check
        # TODO is this sql table worth anything if both keys actually need to be
        # referenced later anyway?

        # TODO only add as many as there were blocks from thorsync timing info?
        odor1_ids = [self.db_odors.at[o1,'odor_id'] for o1, _ in odor_pair_list]
        odor2_ids = [self.db_odors.at[o2,'odor_id'] for _, o2 in odor_pair_list]

        # TODO TODO make unique first. only need order for filling in the values
        # in responses.
        mixtures = pd.DataFrame({
            'odor1': odor1_ids,
            'odor2': odor2_ids
        })
        # TODO maybe defer this to accepting...
        u.to_sql_with_duplicates(mixtures, 'mixtures')

        frame_times = np.array(ti['frame_times']).flatten()

        # Frame indices for CNMF output.
        # Of length equal to number of blocks. Each element is the frame
        # index (from 1) in CNMF output that starts the block, where
        # block is defined as a period of continuous acquisition.
        block_first_frames = np.array(ti['trial_start'], dtype=np.uint32
            ).flatten() - 1

        # TODO after better understanding where trial_start comes from,
        # could get rid of this check if it's just tautological
        block_ic_thorsync_idx = np.array(ti['block_ic_idx']).flatten()
        assert len(block_ic_thorsync_idx) == len(block_first_frames), \
            'variables in MATLAB ti have inconsistent # of blocks'

        # TODO unit tests for block handling code
        n_blocks_from_gsheet = last_block - first_block + 1
        n_blocks_from_thorsync = len(block_first_frames)

        assert (len(odor_pair_list) == (last_block - first_block + 1) *
            presentations_per_block)

        n_presentations = n_blocks_from_gsheet * presentations_per_block

        err_msg = ('{} blocks ({} to {}, inclusive) in Google sheet {{}} {} ' +
            'blocks from ThorSync.').format(n_blocks_from_gsheet,
            first_block + 1, last_block + 1, n_blocks_from_thorsync)
        fail_msg = (' Fix in Google sheet, turn off ' +
            'cache if necessary, and rerun.')

        allow_gsheet_to_restrict_blocks = True

        # TODO factor all this code out, but especially these checks, so that
        # populate_db would catch this as well
        if n_blocks_from_gsheet > n_blocks_from_thorsync:
            raise ValueError(err_msg.format('>') + fail_msg)

        elif n_blocks_from_gsheet < n_blocks_from_thorsync:
            if allow_gsheet_to_restrict_blocks:
                warnings.warn(err_msg.format('<') + (' This is ONLY ok if you '+
                    'intend to exclude the LAST {} blocks in the Thor output.'
                    ).format(n_blocks_from_thorsync - n_blocks_from_gsheet))
            else:
                raise ValueError(err_msg.format('<') + fail_msg)

        # TODO maybe factor this printing stuff out?
        print(('{} comparisons ({{A, B, A+B}} in random order x {} repeats)')
            .format(n_blocks_from_gsheet, n_repeats))

        # TODO maybe print this in tabular form?
        # TODO TODO TODO use abbreviations (defined in one place for all hong
        # lab code, ideally)
        for i in range(n_blocks_from_gsheet):
            p_start = presentations_per_block * i
            p_end = presentations_per_block * (i + 1)
            cline = '{}: '.format(i)

            odor_strings = []
            for o in odor_pair_list[p_start:p_end]:
                if o[1] == 'paraffin':
                    odor_string = o[0]
                else:
                    odor_string = ' + '.join(o)
                odor_strings.append(odor_string)

            print(cline + ', '.join(odor_strings))
        print('')

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

        # TODO maybe just load a range of movie (if not all blocks/frames used)?

        ########################################################################
        # End stuff more likely to fail
        ########################################################################
        # Need to make sure we don't think the output of CNMF from other data is
        # associated with the new data we load.
        self.cnm = None

        self.odor_onset_frames = np.array(ti['stim_on'], dtype=np.uint32
            ).flatten() - 1
        self.odor_offset_frames = np.array(ti['stim_off'], dtype=np.uint32
            ).flatten() - 1

        assert len(self.odor_onset_frames) == len(self.odor_offset_frames)

        block_last_frames = np.array(ti['trial_end'], dtype=np.uint32
            ).flatten() - 1

        if allow_gsheet_to_restrict_blocks:
            # TODO unit test for case where first_block != 0 and == 0
            # w/ last_block == first_block and > first_block
            block_first_frames = block_first_frames[
                :(last_block - first_block + 1)]
            block_last_frames = block_last_frames[
                :(last_block - first_block + 1)]

            assert len(block_first_frames) == n_blocks_from_gsheet
            assert len(block_last_frames) == n_blocks_from_gsheet

            self.odor_onset_frames = self.odor_onset_frames[
                :(last_presentation - first_presentation + 1)]

            self.odor_offset_frames = self.odor_offset_frames[
                :(last_presentation - first_presentation + 1)]

            assert len(self.odor_onset_frames) == n_presentations
            assert len(self.odor_offset_frames) == n_presentations

            frame_times = frame_times[:(block_last_frames[-1] + 1)]

        assert len(self.odor_onset_frames) == len(odor_pair_list)

        last_frame = block_last_frames[-1]

        n_tossed_frames = movie.shape[0] - (last_frame + 1)
        if n_tossed_frames != 0:
            print(('Tossing trailing {} of {} frames of movie, which did not ' +
                'belong to any used block.').format(
                n_tossed_frames, movie.shape[0]))

        # TODO want / need to do more than just slice to free up memory from
        # other pixels? is that operation worth it?
        self.movie = movie[:(last_frame + 1)]
        assert self.movie.shape[0] == len(frame_times), \
            '{} != {}'.format(self.movie.shape[0], len(frame_times))

        self.date = date
        self.fly_num = fly_num
        self.thorimage_id = thorimage_id
        self.started_at = started_at
        self.n_repeats = n_repeats
        self.n_blocks = n_blocks_from_gsheet
        self.presentations_per_repeat = presentations_per_repeat
        self.presentations_per_block = presentations_per_block 
        self.odor1_ids = odor1_ids
        self.odor2_ids = odor2_ids
        self.frame_times = frame_times
        self.block_first_frames = block_first_frames

        self.matfile = mat

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
        # (would want to do it before slicing probably?)
        self.tiff_md5 = md5(tiff)
        end = time.time()
        print('Hashing TIFF took {:.3f} seconds'.format(end - start))

        self.tiff_mtime = datetime.fromtimestamp(getmtime(tiff))

        self.start_frame = None
        self.stop_frame = None

        self.fig.clear()
        # TODO maybe add some height for pixel based correlation matrix if i
        # include that
        fig_w_inches = 7
        fig_h_inches = 7
        self.set_fig_size(fig_w_inches, fig_h_inches)

        ax = self.fig.add_subplot(111)
        # TODO maybe replace w/ pyqtgraph video viewer roi, s.t. it can be
        # restricted to a different ROI if that would help
        ax.plot(np.mean(self.movie, axis=(1,2)))
        ax.set_title(self.recording_title)
        self.fig.tight_layout()
        self.mpl_canvas.draw()

        self.run_cnmf_btn.setEnabled(True)
        self.accept_cnmf_btn.setEnabled(False)
        self.reject_cnmf_btn.setEnabled(False)
        self.accepted = None
        self.uploaded_presentations = False


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

        # TODO TODO maybe just read presentations in here if this is the only
        # widget that's going to use it? (want it to be up-to-date anyway...)

        self.presentations = presentations.set_index(comp_cols)
        # TODO same as w/ presentations. maybe not global.
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

        self.fig = Figure()
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

        tiff = u.motion_corrected_tiff_filename(*comp[:-1])
        # TODO move loading to some other process? QRunnable? progress bar?
        # TODO if not just going to load just comparison, keep movie loaded if
        # clicking other comparisons / until run out of memory
        # TODO just load part of movie for this comparison
        print('Loading TIFF {}...'.format(tiff), end='')
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
        fps = u.fps_from_thor(self.metadata)
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
            u.crop_to_nonzero(footprint, margin=6)

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
                        u.closed_mpl_contours(f, ax, colors='blue')

                contour = \
                    u.closed_mpl_contours(cropped_footprint, ax, colors='red')
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
            pd.DataFrame({'nickname': [self.user]}).set_index('nickname'
                ).to_sql('people', u.conn, if_exists='append',
                method=u.pg_upsert)

            user_select.addItem(self.user)

        for nickname in self.nicknames:
            if nickname != self.user:
                user_select.addItem(nickname)

        user_select.setEditable(True)
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


        # TODO add this back after fixing size policies s.t.
        # buttons aren't cut off at the bottom.
        '''
        debug_action = QAction('&Debug shell', self)
        debug_action.triggered.connect(self.debug_shell)

        menu = self.menuBar()
        tools = menu.addMenu('&Tools')
        tools.addAction(debug_action)
        '''
        debug_shortcut = QShortcut(QKeySequence('Ctrl+d'), self)
        debug_shortcut.activated.connect(self.debug_shell)

        self.setFocus()

    
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


    def debug_shell(self):
        import ipdb
        ipdb.set_trace()


def main():
    global presentations
    global recordings
    global recordings_meta
    global footprints
    global comp_cols
    # TODO try to eliminate matlab requirement here. currently just to read
    # matlab "ti" (timing info) output from Remy's cnmf MAT files
    # (but maybe h5py worked in that case?)
    global evil
    global common_code_versions

    # Calling this first to minimize chances of code diverging.
    # TODO might be better to do before some imports... some of the imports are
    # kinda slow
    # TODO will __file__ still work if i get to the point of installing this
    # package w/ pip?
    common_code_versions = [u.version_info(m,
        used_for='extracting footprints and traces')
        for m in [caiman, __file__]]

    # TODO TODO matlab stuff that only generated saved output needs to be
    # handled, and it can't work this way.

    # TODO TODO TODO maybe use gitpython to check for remote updates and pull
    # them / relaunch / fail until user pulls them?

    # TODO TODO also deal w/ matlab code versionS somehow...

    # TODO maybe rename all of these w/ db_ prefix or something, to
    # differentiate from non-global versions in segmentation tab code
    print('reading odors from postgres...', end='')
    odors = pd.read_sql('odors', u.conn)
    print(' done')

    print('reading presentations from postgres...', end='')
    presentations = pd.read_sql('presentations', u.conn)
    print(' done')

    presentations['from_onset'] = presentations.from_onset.apply(
        lambda x: np.array(x))

    presentations = u.merge_odors(presentations, odors)

    # TODO change sql for recordings table to use thorimage dir + date + fly
    # as index?
    recordings = pd.read_sql('recordings', u.conn)

    presentations = u.merge_recordings(presentations, recordings)

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
    footprints.set_index(u.recording_cols, inplace=True)

    comp_cols = u.recording_cols + ['comparison']

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

