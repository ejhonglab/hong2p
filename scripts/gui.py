#!/usr/bin/env python3

"""
GUI to do make ROIs for trace extraction and validate those ROIs.
"""

import sys
import os
from os.path import split, join, exists, sep, getmtime
import warnings
from collections import defaultdict
from functools import partial
import time
from datetime import datetime
import socket
import getpass
import pickle
from copy import deepcopy
from io import BytesIO
import pprint
import traceback
from shutil import copyfile
import json
import glob

from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot, QThreadPool, QRunnable,
    Qt, QEvent, QVariant)
from PyQt5.QtGui import QColor, QKeySequence, QCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QListWidget,
    QGroupBox, QPushButton, QLineEdit, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QLabel, QListWidgetItem, QScrollArea, QAction, QShortcut,
    QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QFileDialog, QMenu, QMessageBox, QInputDialog, QSizePolicy)
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

# TODO allow things to fail gracefully if we don't have cnmf.
# either don't display tabs that involve it or display a message indicating it
# was not found.
import caiman
from caiman.source_extraction.cnmf import params, cnmf
import caiman.utils.visualization
# Need functions only in my fork of this.
import ijroi
import chemutils as cu

from hong2p import util, matlab, db, thor, viz
from hong2p.roi import (crop_to_nonzero, py2imagej_coords, db_row2footprint,
    db_footprints2array, ijrois2masks, footprints_to_flat_cnmf_dims,
    extract_traces_bool_masks
)


conn = db.get_db_conn()

# TODO probably move to / use something already in util
# Maybe rename. It's these cols once already in a recording + comparison.
cell_cols = ['name1', 'name2', 'repeat_num', 'cell']

raw_data_root = util.raw_data_root()
analysis_output_root = util.analysis_output_root()

use_cached_gsheet = False
show_inferred_paths = True
overwrite_older_analysis = True

#df = util.mb_team_gsheet(use_cache=use_cached_gsheet)


def show_mask_union(masks):
    # TODO either fail in / handle volumetric timeseries case
    all_footprints = np.any(masks, axis=-1).astype(np.uint8) * 255
    cv2.imshow('all_footprints', all_footprints)


# TODO worth allowing selection of a folder?
# TODO worth saving labels to some kind of file (CSV?) in case database not
# reachable? fn to load these to database?

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
        self.motion_corrected_tifs = util.list_motion_corrected_tifs()

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
        # db (in case other programs have edited it) (or just push from db...)
        # TODO get rid of space on left that wasnt there w/ tree widget
        # TODO start this thing scrolled all the way down
        self.data_tree = QTreeWidget(self)
        self.data_tree.setHeaderHidden(True)
        self.data_and_ctrl_layout.addWidget(self.data_tree)
        # TODO look like i want w/o this? (don't want stuff cut off)
        # (it does not. would need to change other size settings)
        #self.data_tree.setFixedWidth(240)
        self.data_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_tree.customContextMenuRequested.connect(self.data_tree_menu)
        self.data_tree.setExpandsOnDoubleClick(False)
        self.current_recording_widget = None
        self.current_segrun_widget = None

        self.accepted_color = '#7fc97f'
        self.partially_accepted_color = '#f9e54a'
        self.rejected_color = '#ff4d4d'

        global recordings
        print('populating data browser...', end='', flush=True)
        label_set = set()
        for d in self.motion_corrected_tifs:
            x = d.split(sep)
            fname_parts = x[-1].split('_')
            cor_type = 'rigid' if fname_parts[-1][:-4] == 'rig' else 'non-rigid'
            del fname_parts

            date_str = x[-4]
            fly_str = x[-3]
            thorimage_id = util.tiff_thorimage_id(d)
            item_parts = [date_str, fly_str, thorimage_id]#, cor_type]

            label = '/'.join(item_parts)
            assert label not in label_set
            label_set.add(label)

            recording_node = QTreeWidgetItem(self.data_tree)
            # 0 for 0th column
            recording_node.setText(0, label)
            self.data_tree.addTopLevelItem(recording_node)

            # Delete if this seems OK. Deferring to expansion since
            # list_segmentations is currently a slow step.
            '''
            tif_seg_runs = db.list_segmentations(d)
            if tif_seg_runs is None:
                continue

            for _, r in tif_seg_runs.iterrows():
                self.add_segrun_widget(recording_node, r)

            self.color_recording_node(recording_node)
            '''
            # TODO TODO some way to color that is faster than call to
            # list_segmentations? (so can do it before loading)
            # TODO TODO TODO also try to find whether there are any segruns
            # to populate in a cheaper way than actually listing them, so we can
            # only show the expansion symbol when there are children
            recording_node.setChildIndicatorPolicy(
                QTreeWidgetItem.ShowIndicator
            )
            recording_node.setData(0, Qt.UserRole, False)

        print(' done')

        # TODO if this works, also resize on each operation that changes the
        # contents
        #self.data_tree.resizeColumnToContents(0)
        # TODO tweak to actually show all the text of children w/o cutoff off
        # or extra space (is this a matter of child size policies not being set
        # correctly?)
        self.data_tree.setSizePolicy(
            QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        )

        # TODO TODO TODO disable this when appropriate (when running cnmf or
        # loading something else) (or just return under those conditions in this
        # callback)
        self.data_tree.itemDoubleClicked.connect(self.handle_treeitem_dblclick)

        self.data_tree.itemExpanded.connect(self.handle_treeitem_expand)

        # TODO should save cnmf output be a button here or to the right?
        # maybe run should also be to the right?
        # TODO maybe a checkbox to save by default or something?

        # TODO maybe include a progress bar here? to the side? below?

        # TODO why is data/decay_time not in temporal parameters?
        self.default_json_params = '.default_cnmf_params.json'
        # TODO maybe search indep of cwd?
        if exists(self.default_json_params):
            print('Loading default parameters from {}'.format(
                self.default_json_params
            ))
            self.params = \
                params.CNMFParams.from_json_file(self.default_json_params)
        else:
            self.params = params.CNMFParams()

        self.param_widget_stack = QStackedWidget(self)
        self.data_params = dict()
        ####self.current_param_tab_name = 'Initialization'
        # TODO TODO copy once here to get params to reset back to in GUI
        self.cnmf_ctrl_widget = self.make_cnmf_param_widget(self.params,
            editable=True)
        self.param_display_widget = None

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

        self.fig = Figure()
        self.mpl_canvas = FigureCanvas(self.fig)
        # TODO this fixed_dpi thing work? let me make things resizable again?
        # unclear... probably no?
        ###self.mpl_canvas.fixed_dpi = 100

        # TODO test that zoom doesn't change what is serialized. especially
        # the png / svgs (it doesnt clip but does change some relative sizes
        # / spaces)
        # TODO maybe a qgraphics(area/view?) would be more responsive / provide
        # same features + allow dragging?
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

        # TODO TODO maybe make display_widget tabbed, w/ one tab as it
        # currently is, and the other to control postprocessing params (like
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
        # TODO further decrease space between btns and checkboxes
        display_btns_layout.setContentsMargins(0, 0, 0, 0)
        display_btns_layout.setSpacing(0)

        self.block_label_btn_widget = display_btns
        self.block_label_btns = []
        self.block_label_btn_layout = display_btns_layout
        self.upload_btn = QPushButton('Upload', display_btns)
        self.upload_btn.clicked.connect(self.upload_cnmf)
        self.display_layout.addWidget(self.upload_btn)

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
        self.plot_intermediates_btn = QCheckBox('Intermediate footprints',
            display_params
        )
        self.plot_intermediates = False
        self.plot_intermediates_btn.setChecked(self.plot_intermediates)
        self.plot_intermediates_btn.stateChanged.connect(partial(
            self.set_boolean, 'plot_intermediates')
        )
        display_params_layout.addWidget(self.plot_intermediates_btn)

        self.plot_corrs_btn = QCheckBox('Correlations', display_params)
        self.plot_correlations = False
        self.plot_corrs_btn.setChecked(self.plot_correlations)
        self.plot_corrs_btn.stateChanged.connect(partial(
            self.set_boolean, 'plot_correlations')
        )
        display_params_layout.addWidget(self.plot_corrs_btn)

        self.plot_traces_btn = QCheckBox('Traces', display_params)
        self.plot_traces = False
        self.plot_traces_btn.setChecked(self.plot_traces)
        self.plot_traces_btn.stateChanged.connect(partial(
            self.set_boolean, 'plot_traces')
        )
        display_params_layout.addWidget(self.plot_traces_btn)

        # TODO TODO warn if would run analysis on same data w/ same params as
        # had previously led to a rejection (actually just same params period)

        # TODO TODO provide the opportunity to compare outputs of sets of
        # parameters, either w/ same displays side by side, or overlayed?

        # TODO TODO maybe some button to automatically pick best set of
        # parameters from database? (just max / some kind of average?)

        # TODO maybe share this across all widget classes?
        self.threadpool = QThreadPool()

        # TODO TODO put all shorcuts in a "tools" menu at least
        self.break_tiff_shortcut = QShortcut(QKeySequence('Ctrl+b'), self)
        self.break_tiff_shortcut.activated.connect(self.save_tiff_blocks)

        self.save_ijrois_shortcut = QShortcut(QKeySequence('Ctrl+r'), self)
        self.save_ijrois_shortcut.activated.connect(self.save_ijrois)
        # TODO TODO maybe store some id (hash?) of generated sets of ROIs, and
        # offer to load any ROI sets in the path that don't match one of these
        # IDs? (without the hash, since i'm currently using mtime for run_at,
        # could just check mtimes)

        self.load_ijrois_shortcut = QShortcut(QKeySequence('Ctrl+f'), self)
        self.load_ijrois_shortcut.activated.connect(self.load_ijrois)

        # TODO shortcut to save cnmf state

        # TODO hide upload_btn by default if that's what i'm gonna do upon
        # opening a recording
        self.upload_btn.setEnabled(False)
        self.tiff_fname = None
        self.movie = None
        self.cnm = None
        self.processing = False
        self.params_changed = False
        self.run_at = None
        self.relabeling_db_segrun = False
        self.ijroi_file_path = None
        self.orig_cnmf_footprints = None

        self.footprint_df = None

        # TODO delete / handle differently
        self.ACTUALLY_UPLOAD = True
        #


    def update_param_tab_index(self, param_tabs) -> None:
        si = self.param_widget_stack.currentIndex()
        if si == -1:
            # -1 is the initial index, before any is set / widgets are added.
            # could set initial param tab i want here if i want.
            # (just set curr_tab_name here and move loop after conditional)
            pass
        else:
            curr_param_tabs = self.param_widget_stack.widget(si).param_tabs
            curr_tab_name = curr_param_tabs.tabText(
                curr_param_tabs.currentIndex()
            )
            for i in range(param_tabs.count()):
                if param_tabs.tabText(i) == curr_tab_name:
                    param_tabs.setCurrentIndex(i)
                    break


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
    # TODO TODO TODO button to make these parameters the current ones
    # (play nice w/ version changes, i.e. the cases above)
    # TODO maybe do the above tab-by-tab as well?
    def make_cnmf_param_widget(self, cnmf_params, editable=False):
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
        hidden_param_json = '.hidden_cnmf_params.json'
        if exists(hidden_param_json):
            with open(hidden_param_json, 'r') as f:
                json_data = json.load(f)
            for g, ps in json_data.items():
                dont_show_by_group[g] = dont_show_by_group[g] | set(ps)
        else:
            json_data = {g: list(ps) for g, ps in dont_show_by_group.items()}
            with open(hidden_param_json, 'w') as f:
                json.dump(json_data, f, indent=2)

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
                desc = ''
                parse_desc = False
                for line in params.CNMFParams.__init__.__doc__.split('\n'):
                    if k + ':' in line:
                        doc_line = line.strip()
                        parse_desc = True
                        continue

                    if parse_desc:
                        stripped = line.strip()
                        if stripped == '':
                            break
                        else:
                            desc += stripped

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

                if g == 'data' and k in self.data_params:
                    v = self.data_params[k]
                    is_data_param = True
                else:
                    is_data_param = False

                if type(v) is bool:
                    # TODO tristate in some cases?
                    w = QCheckBox(group)

                    w.setChecked(v)
                    assert not w.isTristate()

                    if editable and not is_data_param:
                        w.stateChanged.connect(
                            partial(self.cnmf_set_boolean, group_key, k)
                        )

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

                    if editable and not is_data_param:
                        w.valueChanged.connect(
                            partial(self.cnmf_set_from_spinbox, group_key, k)
                        )

                elif type(v) is float:
                    # TODO set step and decimal relative to default size?
                    # (1/10%?)
                    # TODO range?
                    # TODO maybe assume stuff in [0,1] should stay there?
                    # TODO TODO harcode range for some? or is the precision for
                    # something like nrgthr OK to apply to all?
                    w = QDoubleSpinBox(group)

                    float_min = -1.
                    float_max = 10000.
                    w.setRange(float_min, float_max)
                    assert v >= float_min and v <= float_max
                    w.setValue(v)

                    # Maybe set this per-parameter (leave default of 2 and add
                    # more for parameters that require it, like nrgthr)?
                    # TODO as per warning in qt5 docs, check that min/max/value
                    # have not changed (or set this first? that ok?)
                    w.setDecimals(4)

                    if editable and not is_data_param:
                        w.valueChanged.connect(
                            partial(self.cnmf_set_from_spinbox, group_key, k)
                        )

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

                    if editable and not is_data_param:
                        w.currentIndexChanged[str].connect(
                            partial(self.cnmf_set_from_list, group_key, k)
                        )

                else:
                    print_stuff = True
                    w = QLineEdit(group)
                    w.setText(repr(v))
                    if editable and not is_data_param:
                        # TODO TODO if using eval, use setValidator to set some
                        # validator that eval call works?
                        w.editingFinished.connect(
                            partial(self.cnmf_set_from_text, group_key, k, w)
                        )

                if formgen_print and print_stuff:
                    print(k, v, type(v))
                    print(doc_line)
                    print('')

                if not editable or is_data_param:
                    w.setEnabled(False)

                if desc != '':
                    w.setToolTip(desc)

                group_layout.addRow(k, w)

            if formgen_print:
                print('')

        if formgen_print:
            print('Seen types:', seen_types)

        # TODO TODO fix segrun -> segrun case. this does not use correct current
        # tab. may be using current tab for other (editable, idx=0) cnmf widget?
        self.update_param_tab_index(param_tabs)

        # TODO worth doing this patching or is there already a pretty easy
        # way to get param_tabs from returned cnmf_ctrl_widget w/ qt stuff?
        cnmf_ctrl_widget.param_tabs = param_tabs

        cnmf_ctrl_layout = QVBoxLayout(cnmf_ctrl_widget)
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

        if not editable:
            mk_current_params_btn = QPushButton('Use These Parameters')
            param_btns_layout.addWidget(mk_current_params_btn)
            mk_current_params_btn.clicked.connect(lambda:
                self.change_cnmf_params(param_json_str)
            )

        mk_default_params_btn = QPushButton('Make Parameters Default')
        param_btns_layout.addWidget(mk_default_params_btn)
        if editable:
            # Wrapped with lambda to try to prevent qt from adding some
            # bool value to the *args.
            mk_default_params_btn.clicked.connect(lambda:
                self.save_default_params()
            )
        else:
            assert param_json_str is not None
            # TODO have this notify + avoid saving if params are already
            # default (read just before)
            mk_default_params_btn.clicked.connect(lambda:
                self.save_default_params(param_json_str)
            )

        if editable:
            # TODO support this?
            load_params_btn = QPushButton('Load Parameters From File')
            load_params_btn.setEnabled(False)
            param_btns_layout.addWidget(load_params_btn)

        save_params_btn = QPushButton('Save Parameters To File')
        if editable:
            save_params_btn.setEnabled(False)
        else:
            assert param_json_str is not None
            # TODO why didn't lambdas work before again? this case suffer from
            # the same problem?
            save_params_btn.clicked.connect(
                lambda: self.save_json(param_json_str)
            )
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
            # TODO at this point, should i also just make another button for
            # loading ImageJ ROIs? should there even be something of a param
            # widget in that case (for controlling things like detrending...
            # idk) though maybe things i'd want those parameters are things i'd
            # want in all cases. maybe none specific to ijroi case...
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


    # TODO want to fail somewhat gracefully if parameters types/keys are
    # inconsistent w/ old set though..., so maybe want for success to delete?
    def change_cnmf_params(self, json_str) -> None:
        self.param_widget_stack.removeWidget(self.param_display_widget)
        self.param_widget_stack.removeWidget(self.cnmf_ctrl_widget)
        sip.delete(self.cnmf_ctrl_widget)
        self.cnmf_ctrl_widget = self.make_cnmf_param_widget(json_str,
            editable=True
        )
        self.param_widget_stack.addWidget(self.cnmf_ctrl_widget)
        self.param_widget_stack.addWidget(self.param_display_widget)
        self.param_widget_stack.setCurrentIndex(0)


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


    def save_tiff_blocks(self):
        if self.movie is None:
            return

        # TODO reimplement w/ start+end block + new name in same dialog,
        # rather than one popup for each

        first_from1 = self.first_block + 1
        last_from1 = self.last_block + 1
        i_from1, ok_pressed = QInputDialog.getInt(self, 'Start block',
            'Start block ({}-{}):'.format(first_from1, last_from1),
            value=first_from1, min=first_from1, max=last_from1
        )
        # TODO this check right?
        if not ok_pressed:
            return

        j_from1, ok_pressed = QInputDialog.getInt(self, 'End block',
            'End block ({}-{}):'.format(i_from1, last_from1),
            value=i_from1, min=i_from1, max=last_from1
        )
        if not ok_pressed:
            return

        new_first = i_from1 - 1
        # TODO test indexing on this one is right
        new_last = j_from1 - 1

        if self.tiff_fname.endswith('_nr.tif'):
            cor_type = 'nr'
        elif self.tiff_fname.endswith('_rig.tif'):
            cor_type = 'rig'

        new_thorimage_id = '{}_{}b{}_from_{}'.format(self.thorimage_id,
            i_from1, j_from1, cor_type
        )

        date_dir = self.date.strftime('%Y-%m-%d')
        fly_dir = str(self.fly_num)
        new_pathend = join(date_dir, fly_dir, new_thorimage_id)
        fake_raw_dir = join(raw_data_root, new_pathend)
        fake_raw_ts_dir = join(raw_data_root, new_pathend + '_sync')

        print('\nMaking directories:')
        if not exists(fake_raw_dir):
            print(fake_raw_dir)
            os.mkdir(fake_raw_dir)

        if not exists(fake_raw_ts_dir):
            print(fake_raw_ts_dir)
            os.mkdir(fake_raw_ts_dir)

        src_raw_dir = join(raw_data_root, date_dir, fly_dir, self.thorimage_id)

        image_xml_name = 'Experiment.xml'
        src_image_xml = join(src_raw_dir, image_xml_name)
        new_image_xml = join(fake_raw_dir, image_xml_name)
        if not exists(new_image_xml):
            print('\nCopying ThorImage XML:\nfrom =', src_image_xml)
            print('to =', new_image_xml)
            copyfile(src_image_xml, new_image_xml)

        assert len(self.recordings) == 1
        # TODO put this def centrally somewhere?
        ts_xml_name = 'ThorRealTimeDataSettings.xml'
        src_sync_xml = join(self.recordings.iloc[0].thorsync_path,
            ts_xml_name
        )
        new_sync_xml = join(fake_raw_ts_dir, ts_xml_name)
        if not exists(new_sync_xml):
            print('\nCopying ThorSync XML:\nfrom =', src_sync_xml)
            print('to =', new_sync_xml)
            copyfile(src_sync_xml, new_sync_xml)

        analysis_dir = join(analysis_output_root, date_dir, fly_dir)
        src_matfile = join(analysis_dir,
            'cnmf', self.thorimage_id + '_cnmf.mat'
        )
        linked_matfile = join(analysis_dir,
            'cnmf', new_thorimage_id + '_cnmf.mat'
        )
        # TODO this check work w/ symlinks?
        if not exists(linked_matfile):
            print('\nSymlinking to MAT file with timing information:')
            print('from =', src_matfile)
            print('to =', linked_matfile)
            # TODO if this will overwrite, check linked doesnt exist
            os.symlink(src_matfile, linked_matfile)

        print('\nFor Google sheet:')
        print('thorimage_dir =', new_thorimage_id)
        print('thorsync_dir =', split(fake_raw_ts_dir)[-1])
        print('first_block =', i_from1)
        print('last_block =', j_from1)

        # TODO share more code between this and avg...
        tiff_path = join(analysis_dir, 'tif_stacks',
            '{}_{}.tif'.format(new_thorimage_id, cor_type)
        )

        start_frame = self.block_first_frames[new_first]
        end_frame = self.block_last_frames[new_last]
        # TODO check this is ~right. maybe assert
        #print('fraction of frames in subset:',
        #    (end_frame - start_frame) / self.movie.shape[0])

        # TODO TODO factor out this saving into util and call here
        # TODO try other metadata as necessary
        # metadata={'axes': 'TZCYX', 'spacing': 3.1, 'unit': 'um'} for example
        # resolution=(float, float)
        # TODO so is this an old version of tifffile if it seems to say in the
        # docs online that imsave has been renamed to imwrite? upgrade?
        # TODO i remember typing (end_frame + 1) before 4002e4a commit,
        # but now it's not here... so did i just accidentally undo, or
        # did i erroneously add this somewhere it doesn't belong?
        sliced_movie = self.movie[start_frame:(end_frame + 1)]
        if not exists(tiff_path):
            print('\nSaving to TIFF {}...'.format(tiff_path), flush=True,
                end=''
            )
            tifffile.imsave(tiff_path, sliced_movie, imagej=True)
            print(' done\n')

        avg_tiff_path = join(analysis_dir, 'tif_stacks', 'AVG',
            'nonrigid' if cor_type == 'nr' else 'rigid',
            'AVG{}_{}.tif'.format(new_thorimage_id, cor_type)
        )
        if not exists(avg_tiff_path):
            print('\nSaving average TIFF to {}'.format(avg_tiff_path))
            sliced_movie_avg = np.mean(sliced_movie, axis=0)
            tifffile.imsave(avg_tiff_path, sliced_movie_avg, imagej=True)

        # TODO if for some reason i *do* continue doing it this way, rather than
        # refactoring sub-recording handling, probably also automate editing the
        # gsheet and reloading the data_tree
        # TODO TODO also automate breaking into sub-recordings of a max
        # # frames / blocks


    # TODO TODO maybe factor out the core of this fn to a new util fn
    def save_ijrois(self):
        """Saves CNMF footprints to ImageJ compatible ROIs.
        """
        raise NotImplementedError('need to recheck this path for '
            'transpositions converting to ImageJ coordinates'
        )
        if self.footprint_df is None:
            print('No footprints loaded. Run CNMF or load a run.')
            return

        if self.ijroi_file_path is not None:
            assert self.parameter_json is None
            print(('Current footprints were loaded from ImageJ output. '
                'Writing ImageJ ROIs not supported in this case.'
            ))
            return

        row = self.current_segrun_widget.data(0, Qt.UserRole)
        tiff = row.input_filename
        xy, z, _ = thor.get_thorimage_dims_xml(thor.tif2xml_root(tiff))
        if z is None:
            frame_shape = xy
        else:
            frame_shape = xy + (z,)

        # TODO should this popup / take a threshold as an arg?
        # (to threshold float CNMF components)
        self.footprint_df.sort_index(inplace=True)
        ijrois = []
        # Just since, as-is, plot_closed_contours will use current MPL ax,
        # and pyqt5 will apparently make a new figure and show it when this call
        # is done.
        # TODO don't have plt... cant do this. can i get plt? how to
        # close?!?
        #fig = plt.figure()
        fig = Figure()
        for cell_row in self.footprint_df.iterrows():
            cell = cell_row[0]
            # TODO this indicate some other problem?
            footprint = db_row2footprint(cell_row[1], shape=frame_shape)
            # Looking at ijroi source code, it seems Y coordinate is first in
            # input / output arrays.
            footprint = py2imagej_coords(footprint)

            # TODO TODO maybe change this to using cv2 findContours or something
            # TODO make this always verbose when it's applying whatever
            # (take_largest, in this case) if_multiple strategy
            # this doesn't effectively transpose stuff, does it?

            # TODO it kinda seems like i need to dilate the roi to the int
            # boundary, to get them to overlap w/ cnmf util plots...
            # (adding 0.5 to contour before casting seemed to produce a similar
            # problem in opposite direction)
            ij_contour = (viz.plot_closed_contours(footprint,
                if_multiple='take_largest')).astype(np.int16)

            # TODO need to subtract one point or something? need to order?
            # (so as not to duplicate start/end) (seems not to not *fail* as-is
            # but maybe some things aren't working right...)
            # TODO maybe also prefix name w/ analysis run timestamp?
            # or put in roi properties somewhere? other file in zip?
            # (ij seems to just ignore extensions it doesn't expect?)
            # TODO maybe use subpixel resolution settings in ijroi to get it to
            # save exactly as same rois, rather than truncated?
            ijrois.append((str(cell) + '.roi', ij_contour))
        del fig
        #plt.close(fig)
        #import ipdb; ipdb.set_trace()

        # TODO delete.
        #self.before_ijroi_cycle = ijrois2masks(ijrois, frame_shape)
        #

        tiff_dir = split(tiff)[0]
        thorimage_id = util.tiff_thorimage_id(tiff)
        roi_filename = (thorimage_id + self.run_at.strftime('_%Y%m%d_%H%M%S') +
            '_ijroi.zip'
        )
        roi_filepath = join(tiff_dir, roi_filename)

        print('Writing ImageJ ROIs to {}'.format(roi_filepath))
        ijroi.write_polygon_roi_zip(ijrois, roi_filepath)


    # TODO TODO TODO maybe factor out the core of this fn to a new util fn
    def load_ijrois(self):
        # TODO part about docstring about CNMF still accurate?
        """Load ImageJ compatible ROIs and calculates same things calculated
        using CNMF output. Loads to db.

        (in all of the below, excluding GUI specific variables)
        Requires:
        - self.movie
        - self.tiff_fname
        - self.thorimage_id
        - self.trial_start_frames
        - self.odor_onset_frames
        - self.trial_stop_frames
        - self.frame_times (but just in part that saves extra debug info...)
        - Perhaps other things used in `process_segmentation_output`...

        Sets:
        - self.orig_cnmf_footprints
            (how does this differ from footprints again? in terms of both
             content / use elsewhere?)
        - self.footprints
        - self.raw_f
        - self.df_over_f
        - self.ijroi_file_path
        - self.run_at
        - self.parameter_json (to None)
        - self.run_len_seconds (to None)
        - self.cnm (to None)
        - If `process_segmentation_output` sets stuff, that too...
        """
        if self.movie is None:
            print('Load a movie before loading ImageJ ROIs.')
            return

        #def ijroi_segmentation(tiff_fname, movie,
        # TODO maybe just combine w/ open_recording and load movie and
        # everything? or optionally?

        ijroiset_filename = util.tiff_ijroi_filename(self.tiff_fname)

        old_plot_intermediates_val = self.plot_intermediates
        self.plot_intermediates_btn.setChecked(False)
        restore_false = False
        if not (self.plot_correlations or self.plot_traces):
            restore_false = True
            self.plot_corrs_btn.setChecked(True)
            self.plot_traces_btn.setChecked(True)

        ijroi_cnmf_run = None
        fname = split(ijroiset_filename)[-1]
        parts = fname.split('_')
        if len(parts) >= 3:
            date_str = '_'.join(parts[-3:-1])
            try:
                ijroi_cnmf_run = datetime.strptime(date_str,
                    '%Y%m%d_%H%M%S'
                )
            except ValueError:
                pass

        frame_shape = self.movie.shape[1:]
        self.orig_cnmf_footprints = None
        if ijroi_cnmf_run is not None:
            full_timestamps = pd.read_sql_query('''SELECT run_at FROM
                segmentation_runs WHERE date_trunc('second', run_at) = '{}'
            '''.format(pd.Timestamp(ijroi_cnmf_run)), conn).run_at
            assert len(full_timestamps) <= 1

            if len(full_timestamps) == 1:
                ijroi_cnmf_run = full_timestamps[0]
                orig_cnmf_footprints = pd.read_sql_query('''
                    SELECT * FROM cells WHERE segmentation_run = '{}'
                    '''.format(pd.Timestamp(ijroi_cnmf_run)),
                    conn, index_col='cell'
                )
                if len(orig_cnmf_footprints) == 0:
                    warnings.warn(('No footprints found in db for CNMF run {}'
                        ).format(util.format_timestamp(ijroi_cnmf_run))
                    )
                else:
                    self.orig_cnmf_footprints = db_footprints2array(
                        orig_cnmf_footprints, frame_shape
                    )
            else:
                warnings.warn('Segmentation run {} not found in db'.format(
                    util.format_timestamp(ijroi_cnmf_run))
                )

        # TODO display a note about ij params if loading ijroi analysis from db
        # TODO TODO make it so output side of qsplitter gets bigger when this
        # happens, rather than having data browser grow.
        self.param_widget_stack.hide()
        # TODO maybe no point in disabling this button, if widget containing it
        # is going to be hidden...
        self.run_cnmf_btn.setEnabled(False)
        self.plot_intermediates_btn.setEnabled(False)
        self.processing = True
        self.fig.clear()
        self.ijroi_file_path = ijroiset_filename
        ijroiset_mtime = datetime.fromtimestamp(getmtime(ijroiset_filename))
        # TODO see notes in setup.sql. (may) want to use another field for
        # ijroiset mtime.
        self.run_at = ijroiset_mtime
        self.parameter_json = None
        self.run_len_seconds = None
        self.cnm = None

        print('Using ImageJ ROIs from {}'.format(self.ijroi_file_path))
        print('Last modified at {}'.format(util.format_timestamp(self.run_at)))

        ijrois = ijroi.read_roi_zip(ijroiset_filename)

        # TODO remove duplicate points in the thing that generates the contours
        # (mpl code) / use diff code to generate them. no point always storing
        # this extra stuff if cv2 doesn't need it...

        self.footprints = ijrois2masks(ijrois, frame_shape)
        if self.orig_cnmf_footprints is not None:
            assert self.footprints.shape == self.orig_cnmf_footprints.shape

        # TODO delete
        '''
        show_mask_union(self.footprints)
        avg = np.mean(self.movie, axis=0)
        avg = avg - np.min(avg)
        avg = avg / np.max(avg)
        cv2.imshow('avg', avg)
        '''

        # TODO try plotting cv2 drawContours for all against input
        # contours? (cnmf or stuff passed through ij? former i guess...)

        # TODO superimpose on avg or something before calculating everything
        # (separate user action to confirm)?

        self.raw_f = extract_traces_bool_masks(self.movie, self.footprints)
        n_footprints = self.raw_f.shape[1]

        # TODO factor df/f calc into util fn
        # TODO TODO maybe factor this into some kind of util fn that applies
        # another fn (perhaps inplace, perhaps onto new array) to each
        # (cell, block) (or maybe just each block, if smooth can be vectorized,
        # so it could also apply in frame-shape-preserved case?)
        '''
        for b_start, b_end in zip(self.block_first_frames,
            self.block_last_frames):

            for c in range(n_footprints):
                # TODO TODO TODO TODO need to be (b_end + 1) since not
                # inclusive? (<-fixed) other problems like this elsewhere?????
                # TODO maybe smooth less now that df/f is being calculated more
                # sensibly...
                self.raw_f[b_start:(b_end + 1), c] = util.smooth_1d(
                    self.raw_f[b_start:(b_end + 1), c], window_len=11)
        '''
        self.df_over_f = np.empty_like(self.raw_f) * np.nan
        for t_start, odor_onset, t_end in zip(self.trial_start_frames,
            self.odor_onset_frames, self.trial_stop_frames):

            # TODO TODO maybe use diff way of calculating baseline
            # (include stuff at end of response? just some percentile over a big
            # window or something?)
            # TODO kwargs to control method of calculating baseline

            # TODO maybe display baseline period on plots for debugging?
            # maybe frame numbers got shifted?
            baselines = np.mean(self.raw_f[t_start:(odor_onset + 1), :], axis=0)
            trial_f = self.raw_f[t_start:(t_end + 1), :]
            self.df_over_f[t_start:(t_end + 1), :] = \
                (trial_f - baselines) / baselines

        # TODO delete
        with open('test_ijroi_extraction.p', 'wb') as f:
            data = {
                'raw_f': self.raw_f,
                'dff': self.df_over_f,
                'footprints': self.footprints,
                'ijrois': ijrois,
                'tiff': self.tiff_fname,
                'fps': thor.get_thorimage_fps_xml(thor.tif2xml_root(
                    self.tiff_fname
                )),
                'odor_onset_frames': self.odor_onset_frames,
                'trial_start_frames': self.trial_start_frames,
                'trial_stop_frames': self.trial_stop_frames,
                'frame_times': self.frame_times,
            }
            pickle.dump(data, f)
        #
        self.plot_intermediates_at_fit = False
        self.process_segmentation_output()

        # TODO TODO TODO TODO need to enable upload (+ make sure it works...)

        # TODO TODO TODO ultimately, sanity check trace extraction by comparing
        # cnmf output to stuff from cycling those same cnmf footprints to ijrois
        # and back (w/o modification)
        self.plot_intermediates_btn.setEnabled(True)
        self.param_widget_stack.show()

        self.plot_intermediates_btn.setChecked(old_plot_intermediates_val)
        if restore_false:
            self.plot_corrs_btn.setChecked(False)
            self.plot_traces_btn.setChecked(False)

        # TODO TODO visually indicate in data browser if a segrun node comes
        # from ijrois (bold / italics? pre/suffix? diff color? diff section)


    def delete_segrun(self, treenode) -> None:
        run_at = treenode.data(0, Qt.UserRole).run_at

        confirmation_choice = QMessageBox.question(self, 'Confirm delete',
            'Remove analysis run from {} from database?'.format(
            util.format_timestamp(run_at)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirmation_choice == QMessageBox.Yes:
            # TODO TODO maybe also delete figures? with prompt?
            # that's probably usually the behavior i want... (even w/o prompt)
            print('Deleting analysis run {} from database'.format(run_at))

            sql = "DELETE FROM analysis_runs WHERE run_at = '{}'".format(run_at)
            ret = conn.execute(sql)

            parent = treenode.parent()

            self.data_tree.removeItemWidget(treenode, 0)
            # maybe remove this sip bit... not sure it's necessary
            sip.delete(treenode)
            # alternative would probably be:
            # treenode.deleteLater()
            # see: stackoverflow.com/questions/5899826

            self.color_recording_node(parent)


    # TODO TODO in some db cleanup script, look for all things where the
    # analysis is marked as accepted, yet either the presentations or the
    # responses are missing. prompt for deletion.


    def add_segrun_widget(self, parent, segrun_row) -> None:
        seg_run_item = QTreeWidgetItem(parent)
        seg_run_item.setData(0, Qt.UserRole, segrun_row)
        seg_run_item.setText(0, util.format_timestamp(segrun_row.run_at))
        seg_run_item.setFlags(seg_run_item.flags() ^ Qt.ItemIsUserCheckable)

        # TODO is this really how i want to do it?
        if 'blocks_accepted' not in segrun_row:
            blocks_accepted = db.accepted_blocks(segrun_row.run_at)
            segrun_row['blocks_accepted'] = blocks_accepted

            # TODO maybe change things s.t. i can get rid of propagate flag,
            # and just always have that true?
            self.color_segrun_node(seg_run_item, propagate=False)

        return seg_run_item


    def color_segrun_node(self, segrun_treeitem, propagate=True) -> None:
        blocks_accepted = segrun_treeitem.data(0, Qt.UserRole).blocks_accepted
        if all(blocks_accepted):
            color = self.accepted_color
        elif any(blocks_accepted):
            color = self.partially_accepted_color
        else:
            color = self.rejected_color
        # TODO TODO maybe no background color if blocks_accepted are all None?
        # right now i'm just trying to not call when want no color

        segrun_treeitem.setBackground(0, QColor(color))
        # TODO maybe get rid of propagate flag
        if propagate:
            self.color_recording_node(segrun_treeitem.parent())


    # TODO after deleting all segruns under a recording node,
    # it should go back to default color
    def color_recording_node(self, recording_widget) -> None:
        any_all_accepted = False
        any_partially_accepted = False
        for i in range(recording_widget.childCount()):
            blocks_accepted = recording_widget.child(i).data(0,
                Qt.UserRole).blocks_accepted

            if all(blocks_accepted):
                any_all_accepted = True
                break
            elif any(blocks_accepted):
                any_partially_accepted = True

        if any_all_accepted:
            recording_widget.setBackground(0, QColor(self.accepted_color))
        elif any_partially_accepted:
            recording_widget.setBackground(0,
                QColor(self.partially_accepted_color)
            )
        else:
            recording_widget.setBackground(0, QColor(self.rejected_color))
            # TODO maybe at least color this way when no children?
            # got from: stackoverflow.com/questions/37761002
            # if i want top level nodes to just have original lack of color
            #recording_widget.setData(0, Qt.BackgroundRole, None)


    '''
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
        ret = conn.execute(sql)

        self.mark_canonical(treenode, True)
    '''


    def data_tree_menu(self, pos):
        node = self.data_tree.itemAt(pos)

        if node is None:
            return

        if node.parent() is not None:
            menu = QMenu(self)
            menutil.addAction('Delete from database',
                lambda: self.delete_segrun(node))

            '''
            any_accepted = any(node.data(0, Qt.UserRole).blocks_accepted)
            if any_accepted:
                menutil.addAction('Make canonical',
                    lambda: self.make_canonical(node)
                )
            '''

            menutil.exec_(QCursor.pos())
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
        if (not self.processing and self.movie is not None):
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


    # TODO consolidate w/ run_cnmf? (or break most stuff set here into another
    # fn also called in load_ijrois case?)
    def start_cnmf_worker(self) -> None:
        self.run_cnmf_btn.setEnabled(False)

        # Might consider moving this to end of cnmf call, so judgement can be
        # made while something else is running, but that might be kind of risky.

        # TODO notify user this is happening (and how?)? checkbox to not do
        # this?  and if checkbox is unticked, just store in db w/ accept as
        # null?
        # TODO TODO TODO reject all blocks (if not labelled)
        '''
        if self.accepted is None and self.cnm is not None:
            self.reject_cnmf()
        '''

        # TODO separate button to cancel? change run-button to cancel?

        # TODO maybe don't do this (yet)?
        # TODO TODO need to call show / update or something? it doesn't seem to
        # be doing this... (test)?
        self.fig.clear()

        # TODO what kind of (if any) limitations are there on the extent to
        # which data can be shared across threads? can the worker modify
        # properties under self, and have those changes reflected here?

        # Pass the function to execute
        # Any other args, kwargs are passed to the run function
        worker = Worker(self.run_cnmf)
        # TODO so how does it know to pass one arg in this case?
        # (as opposed to cnmf_done case)
        worker.signals.result.connect(self.process_segmentation_output)
        # TODO TODO implement. may require allowing callbacks to be passed into
        # cnmf code to report progress?
        #worker.signals.progress.connect(self.progress_fn)

        self.parameter_json = self.params.to_json()
        self.ijroi_file_path = None
        self.cnmf_start_seconds = time.time()
        self.run_at = datetime.fromtimestamp(self.cnmf_start_seconds)

        self.params_changed = False
        self.processing = True

        self.threadpool.start(worker)


    def run_cnmf(self) -> None:
        print('Running CNMF ({})'.format(util.format_timestamp(self.run_at)),
            flush=True
        )
        # TODO TODO TODO if i'm going to null self.cnm in imagej thing, should
        # also probably null any ijroi related instance variables here
        # (more important too, as those may be directly uploaded)

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

        self.plot_intermediates_btn.setEnabled(False)
        self.plot_intermediates_at_fit = self.plot_intermediates
        try:
            # From CNMF docs, about first arg to fit:
            # "images : mapped np.ndarray of shape (t,x,y[,z])"
            # (thought it doesn't actually need to be a memory mapped file)
            self.cnm.fit(self.movie,
                intermediate_footprints=self.plot_intermediates
            )

        # TODO test this case
        except MemoryError as err:
            traceback.print_exc()
            # TODO maybe log this / print traceback regardless
            # TODO get cnmf output not handling this appropriately. fix.
            self.cnm = None
            self.run_cnmf_btn.setEnabled(True)
            # TODO was this not actually printing? cnm was none...
            raise

        finally:
            self.plot_intermediates_btn.setEnabled(True)

        # TODO TODO TODO (actually restrict components to those passing
        # thresholds)
        #self.cnm.estimates.evaluate_components(self.movie, self.cnm.params)
        #import ipdb; ipdb.set_trace()
        #

        # TODO delete this hack after getting cnmf tools for evaluating /
        # filtering components to actually work...
        n_pixels = (self.cnm.estimates.A > 0).sum(axis=0)

        min_component_pixels = self.params_copy.get('quality',
            'min_component_pixels')
        max_component_pixels = self.params_copy.get('quality',
            'max_component_pixels')

        big_enough = (n_pixels >= min_component_pixels)
        small_enough = (n_pixels <= max_component_pixels)
        keep = (big_enough & small_enough)

        n_too_small = (~ big_enough).sum()
        if n_too_small > 0:
            print('Tossing {} components that had less than {} pixels'.format(
                n_too_small, min_component_pixels
            ))
        else:
            print('No components had less than minimum {} pixels'.format(
                min_component_pixels
            ))

        n_too_big = (~ small_enough).sum()
        if n_too_big > 0:
            print(('Tossing {} components that had more than {} pixels'
                ).format(n_too_big, max_component_pixels
            ))
        else:
            print('No components had more than maximum {} pixels'.format(
                max_component_pixels
            ))

        # TODO TODO TODO renumber components within the remaining ones!!!
        import ipdb; ipdb.set_trace()
        # Could use CNMF select_components for this, but it seems poorly
        # implemented, and I'm not sure I want to be able to restore components
        # anyway.
        self.cnm.estimates.A = self.cnm.estimates.A[:, keep]
        #

        # TODO see which parameters are changed?
        # (and is this just a big false negative now?)
        if err_if_cnmf_changes_params:
            assert self.params_copy == self.params, 'CNMF changed params in fit'

        # TODO maybe have a widget that shows the text output from cnmf?

        # TODO TODO TODO w/ python version of cnmf, as i'm using it, do i need
        # to explicitly order components to get same ordering as in matlab ver?
        # (yes, it seems so. seems evaluate_components as it least one thing
        # that does this, and that's not working for me at the moment)

        # TODO do all footprints stuff w/o converting to full array?
        # x,y,n_footprints
        self.footprints = self.cnm.estimates.A.toarray()
        # Assuming equal number of pixels along both dimensions.
        pixels_per_side = int(np.sqrt(self.footprints.shape[0]))
        n_footprints = footprints.shape[-1]
        # TODO actually, is this correct? (i.e. if calculating my own
        # projections of the movie onto these footprints, would i get similar
        # results / do the movie and footprint coords match up? or is it only
        # a function of how i'm plotting things that it matches up, and indexing
        # as normal they do not?)
        # TODO TODO TODO maybe now this will need to be changed, since i've
        # untransposed x_coords and y_coords in the db?
        self.footprints = np.reshape(self.footprints,
            (pixels_per_side, pixels_per_side, n_footprints)
        )

        # TODO could maybe compute my own df/f from this if i'm worried...
        # frame number, cell -> value
        # TODO TODO TODO and how is raw_f diff from extract_... -> diff
        # (shouldn't it be exactly the same? just calc that way, rather than
        # extract_... or whatever?)
        self.raw_f = self.cnm.estimates.C.T

        # TODO TODO TODO to copy what Remy's matlab script does, need to detrend
        # within each "block" (probably want w/in trial, even?)
        # (in which case, might need to mix this and logic in get_recording_dfs)
        if self.cnm.estimates.F_dff is None:
            # quantileMin=8, frames_window=500, flag_auto=True, use_fast=False,
            # (a, b, C, f, YrA)
            # TODO TODO TODO don't i want to use extract_... though, since more
            # exact?
            self.cnm.estimates.detrend_df_f()

        self.df_over_f = self.cnm.estimates.F_dff.T

        self.run_len_seconds = time.time() - self.cnmf_start_seconds
        print('CNMF took {:.1f}s'.format(self.run_len_seconds))


    # TODO TODO TODO maybe factor out the core of this fn to util
    # (maybe df_over_f should only be computed after we can already break the
    # raw_f into blocks appropriately? then being 1 shorter for each block as
    # well)
    # TODO TODO TODO so maybe move df_over_f calculation into this fn?
    # TODO actually provide a way to only initialize cnmf, to test out various
    # initialization procedures (though only_init arg doesn't actually seem to
    # accomplish this correctly) (why was that something this fn should do? was
    # it?)
    def get_recording_dfs(self) -> None:
        """
        Requires:
        - self.raw_f
        - self.df_over_f
        - self.footprints

        (these i haven't yet checked whether they are basically only used here)
        - self.date
        - self.fly_num
        - self.started_at
        - self.run_at

        - self.frame_times
        - self.odor_onset_frames
        - self.odor_offset_frames (this requirement could probably be removed.
              only used in some assertions.)

        - self.presentations_per_block
        - self.pair_case

        - self.odor_ids

        Sets:
        - self.presentation_dfs (list # trials long)
        - self.comparison_dfs (list # trials long)
        - self.footprint_df
        """
        # TODO option to return lists as above / concatenated df?
        # or just switch to returning concatenated df?
        # TODO should i also just merge the meta and response dfs by default?
        # TODO rename all presentation_df(s) to meta_df(s) and
        # comparison_df(s) to response_df(s), if not just gonna merge
        #meta_dfs, response_dfs, footprint_df = util.recording_frames(raw_f,
        #    df_over_f, footprints,

        # TODO are self.df_over_f and raw_f the same shape? shouldn't raw_f be
        # one shorter? otherwise, how is that final value imputed?
        # (put requirement on their relative length in the docstring)
        # don't i have some asserts checking their lengths against each other
        # somewhere?
        n_frames, n_cells = self.df_over_f.shape
        # would have to pass footprints back / read from sql / read # from sql
        #assert n_cells == n_footprints
        # TODO bring back after fixing this indexing issue,
        # whatever it is. as with other check in open_recording
        # (mostly redundant w/ assert comparing movie frames and frame_times in
        # end of open_recording...)
        #assert self.frame_times.shape[0] == n_frames

        self.presentation_dfs = []
        self.comparison_dfs = []
        comparison_num = -1

        # TODO consider deleting this conditional if i'm not actually going to
        # support else case (not used now, see where repeat_num is set in loop)
        if self.pair_case:
            repeats_across_real_blocks = False
        else:
            repeats_across_real_blocks = True
            repeat_nums = {id_group: 0 for id_group in self.odor_ids}

        print('processing presentations...', end='', flush=True)
        for i in range(len(self.trial_start_frames)):
            if i % self.presentations_per_block == 0:
                comparison_num += 1
                if not repeats_across_real_blocks:
                    repeat_nums = {id_group: 0 for id_group in self.odor_ids}

            start_frame = self.trial_start_frames[i]
            stop_frame = self.trial_stop_frames[i]
            onset_frame = self.odor_onset_frames[i]
            offset_frame = self.odor_offset_frames[i]

            assert start_frame < onset_frame
            assert onset_frame < offset_frame
            assert offset_frame < stop_frame

            # If either of these is out of bounds, presentation_frametimes will
            # just be shorter than it should be, but it would not immediately
            # make itself apparent as an error.
            assert start_frame < len(self.frame_times)
            assert stop_frame < len(self.frame_times)

            onset_time = self.frame_times[onset_frame]
            # TODO TODO check these don't jump around b/c discontinuities
            # TODO TODO TODO honestly, i forget now, have i ever had acquisition
            # stop any time other than between "blocks"? do i want to stick to
            # that definition?
            # if it did only ever stop between blocks, i suppose i'm gonna have
            # to paint frames between trials within a block as belonging to one
            # trial or the other, for purposes here...
            presentation_frametimes = \
                self.frame_times[start_frame:stop_frame] - onset_time

            curr_odor_ids = self.odor_ids[i]
            # TODO update if odor ids are ever actually allowed to be arbitrary
            # len list (and not just forced to be length-2 as they are now, b/c
            # of the db mixture table design)
            odor1, odor2 = curr_odor_ids
            #

            if self.pair_case:
                repeat_num = repeat_nums[curr_odor_ids]
                repeat_nums[curr_odor_ids] = repeat_num + 1

            # See note in missing odor handling portion of
            # process_segmentation_output to see reasoning behind this choice.
            else:
                repeat_num = comparison_num

            # TODO check that all frames go somewhere and that frames aren't
            # given to two presentations. check they stay w/in block boundaries.
            # (they don't right now. fix!)

            # TODO share more of this w/ dataframe creation below, unless that
            # table is changed to just reference presentation table
            presentation = pd.DataFrame({
                # TODO fix hack (what was this for again? it really a problem?)
                'temp_presentation_id': [i],
                'prep_date': [self.date],
                'fly_num': self.fly_num,
                'recording_from': self.started_at,
                'analysis': self.run_at,
                # TODO get rid of this hack after fixing earlier association of
                # blocks / repeats (or fixing block structure for future
                # recordings)
                'comparison': comparison_num if self.pair_case else 0,
                'real_block': comparison_num,
                'odor1': odor1,
                'odor2': odor2,
                #'repeat_num': repeat_num if self.pair_case else comparison_num,
                'repeat_num': repeat_num,
                'odor_onset_frame': onset_frame,
                'odor_offset_frame': offset_frame,
                'from_onset': [[float(x) for x in presentation_frametimes]],
                # TODO TODO is this still true?
                # They start as True, since the way I'm doing it now, they
                # aren't uploaded otherwise. Once in db, this can be changed to
                # False.
                'presentation_accepted': True
            })

            # TODO TODO assert that len(presentation_frametimes)
            # == stop_frame - start_frame (off-by-one?)
            # TODO (it would fail now) fix!!
            # maybe this is a failure to merge correctly later???
            # b/c presentation frametimes seems to be defined to be same length
            # above... same indices...
            # (unless maybe self.frame_times is sometimes shorter than
            # self.df_over_f, etc)

            '''
            presentation_dff = self.df_over_f[start_frame:stop_frame, :]
            presentation_raw_f = self.raw_f[start_frame:stop_frame, :]
            '''
            # TODO TODO fix / delete hack!!
            # TODO probably just need to more correctly calculate stop_frame?
            # (or could also try expanding frametimes to include that...)
            actual_frametimes_slice_len = len(presentation_frametimes)
            stop_frame = start_frame + actual_frametimes_slice_len
            presentation_dff = self.df_over_f[start_frame:stop_frame, :]
            presentation_raw_f = self.raw_f[start_frame:stop_frame, :]

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
        print(' done', flush=True)

        n_footprints = self.footprints.shape[-1]
        footprint_dfs = []
        for cell_num in range(n_footprints):
            # TODO could use tuple of slice objects to accomodate arbitrary dims
            # here (x,y,Z). change all places like this.
            sparse = coo_matrix(self.footprints[:,:,cell_num])
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
                # TODO TODO TODO TODO was sparse.col for x_* and sparse.row for
                # y_*. I think this was why I needed to tranpose footprints
                # sometimes. fix everywhere.
                'x_coords': [[int(x) for x in sparse.row.astype('int16')]],
                'y_coords': [[int(x) for x in sparse.col.astype('int16')]],
                'weights': [[float(x) for x in sparse.data.astype('float32')]]
            }))
        self.footprint_df = pd.concat(footprint_dfs, ignore_index=True)


    def plot_footprints(self, contour_axes, plot_intermediates, only_init):
        # TODO TODO allow toggling between type of background image shown
        # (radio / combobox for avg, snr, etc? "local correlations"?)
        # TODO TODO use histogram equalized avg image as one option
        img = self.avg

        n_footprint_axes = len(contour_axes)
        for i in range(n_footprint_axes):
            contour_ax = contour_axes[i]
            contour_ax.axis('off')

            # TODO TODO make callbacks for each step and plot as they become
            # available
            # TODO might also be nice to get self.cnm out of here, if possible.
            # maybe have some std interface for fitting fns to output
            # intermediate graphical debugging info?
            if plot_intermediates:
                if i == 0:
                    # TODO why are title's not working? need axis='on'?
                    # just turn off other parts of axis?
                    contour_ax.set_title('Initialization')
                    A = self.cnm.A_init
                elif i == 1:
                    # TODO TODO also plot the values of these for other
                    # iterations, if available?
                    contour_ax.set_title('After spatial update')
                    A = self.cnm.A_spatial_update_k[0]
                elif i == 2:
                    contour_ax.set_title('After merging')
                    A = self.cnm.A_after_merge_k[0]
                #elif i == 3:
                #    A = self.cnm.A_spatial_refinement_k[0]

            if self.ijroi_file_path is not None:
                assert self.parameter_json is None
                A = footprints_to_flat_cnmf_dims(self.footprints)

                # TODO delete
                '''
                try:
                    bA = util.footprints_to_flat_cnmf_dims(
                        self.before_ijroi_cycle
                    )
                    caiman.utils.visualization.plot_contours(bA, img,
                        ax=contour_ax, display_numbers=False, colors='y',
                        linewidth=1.5
                    )
                except AttributeError:
                    pass
                '''
                #
                if self.orig_cnmf_footprints is not None:
                    orig_A = util.footprints_to_flat_cnmf_dims(
                        self.orig_cnmf_footprints
                    )
                    caiman.utils.visualization.plot_contours(orig_A, img,
                        ax=contour_ax, display_numbers=False, colors='g',
                        linewidth=1.0
                    )
            else:
                A = self.cnm.estimates.A

            # TODO maybe show self.cnm.A_spatial_refinement_k[0] too in
            # plot_intermediates case? should be same though (though maybe one
            # is put back in original, non-sliced, coordinates?)
            if i == n_footprint_axes - 1 and not only_init:
                if self.ijroi_file_path is not None:
                    title = 'ImageJ ROIs from {}'.format(
                        split(self.ijroi_file_path)[-1]
                    )
                else:
                    title = 'Final estimate'
                contour_ax.set_title(title)

            elif not plot_intermediates and i == 0 and only_init:
                contour_ax.set_title('Initialization')

            caiman.utils.visualization.plot_contours(A, img, ax=contour_ax,
                display_numbers=False, colors='r', linewidth=1.0
            )
            # TODO TODO TODO also call evaluate_components and include that in
            # Final estimate (maybe w/ a separate subplot to show what gets
            # filtered?)

            self.mpl_canvas.draw()


    # TODO some kind of test option / w/ test data for this part that doesn't
    # require actually running cnmf
    # TODO make this fn not block gui? did cnmf_done not block gui? maybe that's
    # a reason to keep it? or put most of this stuff in the worker?
    def process_segmentation_output(self) -> None:
        # TODO list which instance variables are required / which are set
        """
        """
        # TODO maybe time this too? (probably don't store in db tho)
        # at temporarily, to see which parts are taking so long...
        print('\nProcessing the output...')

        self.display_params_editable(False)

        plot_intermediates = self.plot_intermediates_at_fit
        plot_correlations = self.plot_correlations
        plot_traces = self.plot_traces

        if self.ijroi_file_path is not None:
            assert self.parameter_json is None
            only_init = False
        else:
            only_init = self.params_copy.get('patch', 'only_init')

        n_footprint_axes = 4 if plot_intermediates and not only_init else 1

        w_inches_footprint_axes = 3
        h_inches_per_footprint_ax = 3
        h_inches_footprint_axes = h_inches_per_footprint_ax * n_footprint_axes

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

        if plot_correlations or plot_traces:
            plot_odor_abbreviation_key = True
            key_rows = 1
        else:
            plot_odor_abbreviation_key = False
            key_rows = 0

        gs_rows = sum([footprint_rows, corr_rows, trace_rows, key_rows])
        gs = self.fig.add_gridspec(gs_rows, 1, hspace=0.4, wspace=0.05)

        # TODO maybe redo gs... maybe it should be derived from the
        # h/w_inches stuff? (if everything gets 2/6 why not just 1/3...?)
        footprint_slice = gs[:2, :]

        footprint_gs = footprint_slice.subgridspec(
            n_footprint_axes, 1, hspace=0, wspace=0
        )

        axs = []
        ax0 = None
        for i in range(footprint_gs._nrows):
            if ax0 is None:
                ax = self.fig.add_subplot(footprint_gs[i])
            else:
                ax = self.fig.add_subplot(footprint_gs[i],
                    sharex=ax0, sharey=ax0
                )
            axs.append(ax)

        contour_axes = np.array(axs)
        self.plot_footprints(contour_axes, plot_intermediates, only_init)

        ###################################################################
        # TODO defer this as much as possible
        # (move later, don't compute traces if not needed for plots / accept)
        self.get_recording_dfs()

        presentations_df = pd.concat(self.presentation_dfs, ignore_index=True)

        # TODO TODO TODO probably just fix self.n_blocks earlier
        # in supermixture case
        # (so there is only one button for accepting and stuff...)
        if self.pair_case:
            n_blocks = self.n_blocks
            presentations_per_block = self.presentations_per_block
        else:
            # TODO delete (though check commented and new are equiv on all
            # non self.pair_case experiments)
            '''
            n_blocks = presentations_df.comparison.max() + 1
            n_repeats = util.n_expected_repeats(presentations_df)
            n_stim = len(presentations_df[['odor1','odor2']].drop_duplicates())
            presentations_per_block = n_stim * n_repeats
            '''
            #
            n_blocks = 1
            presentations_per_block = len(self.odor_ids)

        w_inches_per_corr = 3
        h_inches_corrs = 2 * w_inches_per_corr
        w_inches_per_traceplot = 8
        h_inches_traceplots = 10
        w_inches_corr = w_inches_per_corr * n_blocks
        w_inches_traceplots = w_inches_per_traceplot * n_blocks

        # TODO could set this based on whether i want 1 / both orders
        #h_inches_per_traceplot = 5
        #h_inches_traceplots = h_inches_per_traceplot *

        widths = [w_inches_footprint_axes]
        heights = [h_inches_footprint_axes]
        # TODO can you change fig size after plotting?
        # (right now, calling after footprint stuff)
        if plot_correlations:
            widths.append(w_inches_corr)
            heights.append(h_inches_corrs)

        if plot_traces:
            widths.append(w_inches_traceplots)
            heights.append(h_inches_traceplots)

        fig_w_inches = max(widths)
        fig_h_inches = sum(heights)

        self.set_fig_size(fig_w_inches, fig_h_inches)

        if plot_correlations:
            # (end slice is not included, as always, so it's same size as above)
            corr_slice = gs[2:4, :]
            # 2 rows: one for correlation matrices ordered as in experiment,
            # and the other for matrices ordered by odor
            corr_gs = corr_slice.subgridspec(2, n_blocks,
                hspace=0.4, wspace=0.1
            )
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
            all_blocks_trace_gs = gs[4:-1, :].subgridspec(trace_rows,
                n_blocks, hspace=0.3, wspace=0.15
            )

        if plot_odor_abbreviation_key:
            abbrev_key_ax = self.fig.add_subplot(gs[-1, :])

        if plot_correlations or plot_traces:
            # TODO TODO TODO make this configurable in gui / have correlations
            # update (maybe not alongside CNMF parameters, to avoid confusion?)
            response_calling_s = 3.0

            # TODO check again that this also works in case where odors from
            # this experiment are new (weren't in db before)
            # (and maybe support some local analysis anyway that doesn't require
            # rt through the db...)
            presentations_df = db.merge_odors(presentations_df,
                self.db_odors.reset_index()
            )
            # TODO maybe adapt to case where name2 might have only occurence of
            # an odor, or name1 might be paraffin.
            # TODO TODO check this is actually in the order i want across blocks
            # (idk if name1,name2 are sorted / re-ordered somewhere)
            name1_unique = presentations_df.name1.unique()
            name2_unique = presentations_df.name2.unique()
            # TODO should fail earlier (rather than having to wait for cnmf
            # to finish)
            assert (set(name2_unique) == {NO_ODOR} or
                set(name2_unique) - set(name1_unique) == {'paraffin'}
            )
            # TODO TODO TODO factor all abbreviation into its own function
            # (which transforms dataframe w/ full odor names / ids maybe to
            # df w/ the additional abbreviated col (or renamed col))
            # TODO factor these flags out to be user configurable
            single_letter_abbrevs = False
            abbrev_in_presentation_order = True
            plot_in_presentation_order = True
            if not plot_in_presentation_order:
                # TODO would need to look up fixed desired order, as in
                # kc_mix_analysis, to support this
                raise NotImplementedError

            if single_letter_abbrevs:
                if not abbrev_in_presentation_order:
                    # TODO would (again) need to look up fixed desired order, as
                    # in kc_mix_analysis, to support this
                    raise NotImplementedError
            else:
                if abbrev_in_presentation_order:
                    warnings.warn('abbrev_in_presentation_order can only '
                        'be False if not using single_letter_abbrevs'
                    )

            if not abbrev_in_presentation_order:
                # TODO would (again) need to look up fixed desired order, as
                # in kc_mix_analysis, to support this
                raise NotImplementedError
                # (could implement other order by just reorder name1_unique)

            # TODO probably move this fn from chemutils to hong2p.utils
            odor2abbrev = cu.odor2abbrev_dict(name1_unique,
                single_letter_abbrevs=single_letter_abbrevs
            )

            # TODO rewrite later stuff to avoid need for this.
            # it just adds a bit of confusion at this point.
            # TODO need to deal w/ NO_ODOR in here?
            # So that code detecting which combinations of name1+name2 are
            # monomolecular does not need to change.
            # TODO TODO doesn't chemutils do this at this point? test
            odor2abbrev['paraffin'] = 'paraffin'
            # just so name2 isn't all NaN for now...
            odor2abbrev[NO_ODOR] = NO_ODOR

            block_iter = list(range(n_blocks))
        else:
            block_iter = []

        for i in block_iter:
            # TODO maybe concat and only set whole df as instance variable in
            # get_recording_df? then use just as in kc_analysis all throughout
            # here? (i.e. subset from presentationS_df above...)
            presentation_dfs = self.presentation_dfs[
                (presentations_per_block * i):
                (presentations_per_block * (i + 1))
            ]
            presentation_df = pd.concat(presentation_dfs,
                ignore_index=True
            )
            comparison_dfs = self.comparison_dfs[
                (presentations_per_block * i):
                (presentations_per_block * (i + 1))
            ]

            # Using this in place of NaN, so frame nums will still always have
            # int dtype. maybe NaN would be better though...
            # TODO maybe i want something unlikely to be system dependent
            # though... if i'm ever going to serialize something containing
            # this value...
            INT_NO_REAL_FRAME = sys.maxsize
            last_real_temp_id = presentation_df.temp_presentation_id.max()

            # Not supporting filling in missing odor presentations in pair case
            # (it hasn't happened yet). (and would need to consider within
            # comparisons, since odors be be shared across)
            if not self.pair_case:
                # TODO maybe add an 'actual_block' column or something in
                # paircase? or in both?
                assert presentation_df.comparison.nunique() == 1
                n_full_repeats = presentation_df.odor1.value_counts().max()
                assert len(presentation_df) % n_full_repeats == 0

                odor1_set = set(presentation_df.odor1)
                n_odors = len(odor1_set)

                # n_blocks and n_pres_per_actual_block currently have different
                # meanings in pair_case and not here, w/ blocks in this case not
                # matching actual "scopePin high" blocks.
                # In this case, "actual blocks" have each odor once.
                n_pres_per_actual_block = len(presentation_df) // n_full_repeats

                n_missed_per_block = n_odors - n_pres_per_actual_block

                # TODO TODO add a flag for whether we should fill in missing
                # data like this, and maybe fail if the flag is false and we
                # have missing data (b/c plot labels get screwed up)
                # (don't i already have a flag in open_recording? make more
                # global (or an instance variable)?)

                # TODO may want to assert all odor id lookups work in merge
                # (if it doesn't already functionally do that), because
                # technically it's possible that it just so happens the last
                # odor (the missed one) is always the same

                if n_missed_per_block > 0:
                    # Could modify loop below to iterate over missed odors if
                    # want to support this.
                    assert n_missed_per_block == 1, 'for simplicity'

                    # from_onset is not hashable so nunique on everything fails
                    const_cols = presentation_df.columns[[
                        (False if c == 'from_onset' else
                        presentation_df[c].nunique() == 1)
                        for c in presentation_df.columns
                    ]]
                    const_vals = presentation_df[const_cols].iloc[0].to_dict()

                    # TODO if i were to support where n_cells being different in
                    # each block, would need to subset comparison df to block
                    # and get unique values from there (in loop below)
                    cells = comparison_dfs[0].cell.unique()
                    rec_from = const_vals['recording_from']
                    filler_seg_run = pd.NaT

                    pdf_in_order = \
                        presentation_df.sort_values('odor_onset_frame')

                    next_filler_temp_id = last_real_temp_id + 1
                    for b in range(n_full_repeats):
                        start = b * n_pres_per_actual_block
                        stop = (b + 1) * n_pres_per_actual_block
                        bdf = pdf_in_order[start:stop]

                        row_data = dict(const_vals)
                        row_data['from_onset'] = [np.nan]
                        # Careful! This should be cleared after frame2order def.
                        row_data['odor_onset_frame'] = \
                            bdf.odor_onset_frame.max() + 1
                        row_data['odor_offset_frame'] = INT_NO_REAL_FRAME

                        real_block_nums = bdf.real_block.unique()
                        assert len(real_block_nums) == 1
                        real_block_num = real_block_nums[0]
                        row_data['real_block'] = real_block_num

                        # The question here is whether I want to start the
                        # repeat numbering with presentations that actually have
                        # frames, or whether I want to keep the numbering as it
                        # would have been...

                        # Since in !self.pair_case, real_block num should be
                        # equal to the intended repeat_num.
                        row_data['repeat_num'] = real_block_num
                        # TODO would need to fix this case to handle multiple
                        # missing of one odor, if i did want to have repeat_num
                        # numbering start with presentations that actually have
                        # frames
                        # (- 1 since 0 indexed)
                        #row_data['repeat_num'] = n_full_repeats - 1

                        missing_odor1s = list(odor1_set - set(bdf.odor1))
                        assert len(missing_odor1s) == 1
                        missing_odor1 = missing_odor1s.pop()
                        row_data['odor1'] = missing_odor1

                        row_data['temp_presentation_id'] = next_filler_temp_id

                        presentation_df = \
                            presentation_df.append(row_data, ignore_index=True)

                        # TODO what's the np.nan stuff here for?
                        # why not left to get_recording_dfs?
                        comparison_dfs.append(pd.DataFrame({
                            'temp_presentation_id': next_filler_temp_id,
                            'recording_from': rec_from,
                            'segmentation_run': filler_seg_run,
                            'cell': cells,
                            'raw_f': [[np.nan] for _ in range(len(cells))],
                            'df_over_f': [[np.nan] for _ in range(len(cells))]
                        }))
                        next_filler_temp_id += 1

            frame2order = {f: o for o, f in
                enumerate(sorted(presentation_df.odor_onset_frame.unique()))
            }
            presentation_df['order'] = \
                presentation_df.odor_onset_frame.map(frame2order)
            del frame2order

            # This does nothing if there were no missing odor presentations.
            presentation_df.loc[
                presentation_df.temp_presentation_id > last_real_temp_id,
                'odor_onset_frame'] = INT_NO_REAL_FRAME

            comparison_df = pd.concat(comparison_dfs, ignore_index=True,
                sort=False
            )

            # TODO don't have separate instance variables for presentation_dfs
            # and comparison_dfs if i'm always going to merge here.
            # just merge before and then put in one instance variable.
            # (probably just keep name comparison_dfs)
            presentation_df['from_onset'] = presentation_df['from_onset'].apply(
                lambda x: np.array(x)
            )
            presentation_df = db.merge_odors(presentation_df,
                self.db_odors.reset_index()
            )

            # TODO maybe only abbreviate at end? this approach break upload to
            # database? maybe redo so abbrev only happens before plot?
            # (may want a consistent order across experiments anyway)
            presentation_df['original_name1'] = presentation_df.name1.copy()
            presentation_df['original_name2'] = presentation_df.name2.copy()

            presentation_df['name1'] = \
                presentation_df.name1.map(odor2abbrev)
            presentation_df['name2'] = \
                presentation_df.name2.map(odor2abbrev)

            presentation_df = db.merge_recordings(
                presentation_df, self.recordings
            )

            # TODO TODO TODO assert here, and earlier if necessary, that
            # each odor has all repeat_num + ordering of repeat_num matches
            # that of 'order' column
            #comparison_df[['name1','repeat_num','order']
            #].drop_duplicates().sort_values(['name1','repeat_num','order'])

            # Just including recording_from so it doesn't get duplicated in
            # output (w/ '_x' and '_y' suffixes). This checks recording_from
            # values are all equal, rather than just dropping one.
            # No other columns should be common.
            comparison_df = comparison_df.merge(presentation_df,
                left_on=['recording_from', 'temp_presentation_id'],
                right_on=['recording_from', 'temp_presentation_id']
            )
            comparison_df.drop(columns='temp_presentation_id', inplace=True)
            del presentation_df

            comparison_df = util.expand_array_cols(comparison_df)

            # TODO TODO make this optional
            # (and probably move to upload where fig gets saved.
            # just need to hold onto a ref to comparison_df)
            df_filename = (self.run_at.strftime('%Y%m%d_%H%M_') +
                self.recording_title.replace('/','_') + '.p'
            )
            df_filename = join(analysis_output_root, 'trace_pickles',
                df_filename
            )

            print('writing dataframe to {}...'.format(df_filename), end='',
                flush=True
            )
            # TODO TODO write a dict pointing to this, to also include PID
            # information in another variable?? or at least stuff to index
            # the PID information?
            comparison_df.to_pickle(df_filename)
            print(' done', flush=True)

            # TODO TODO add column mapping odors to order -> sort (index) on
            # that column + repeat_num to order w/ mixture last

            ###################################################################
            if plot_traces or plot_correlations:
                # In plot_traces case above, odor order is handled inside
                # plot_traces, and should give the same answer as here.
                odor_order = None
                if not self.pair_case:
                    # TODO check legend below is also in this order?
                    odor_order = util.df_to_odor_order(comparison_df,
                        return_name1=True)

                # TODO TODO might want to only compute responders/criteria one
                # place, to avoid inconsistencies (so either move this section
                # into next loop and aggregate, or index into this stuff from
                # within that loop?)
                in_response_window = ((comparison_df.from_onset > 0.0) &
                    (comparison_df.from_onset <= response_calling_s)
                )
                # TODO TODO include from_onset col then compute mean?
                window_df = comparison_df.loc[in_response_window,
                    cell_cols + ['order','from_onset','df_over_f']
                ]
                # TODO maybe move this to bottom, around example trace plotting
                window_by_trial = \
                    window_df.groupby(cell_cols + ['order'])['df_over_f']

                window_trial_means = window_by_trial.mean()

            if plot_traces:
                # TODO TODO make this configurable in the gui (may also want
                # default of ~30)
                n = 20

                # TODO TODO TODO plot (at least option for, but prob also
                # default) top cells per odor + provide visual demarcation as to
                # which cells are for which odor

                # TODO TODO TODO uncomment after fixing footprint issue
                # causing footprint_row in util to have > 1 row
                '''
                # TODO or maybe just set show_footprints to false?
                footprints = db.merge_recordings(self.footprint_df,
                    self.recordings
                )
                footprints = db.merge_gsheet(footprints, df)
                footprints.set_index(util.recording_cols + ['cell'],
                    inplace=True
                )
                '''

                # TODO TODO factor out response calling and also do that here,
                # so that random subset can be selected from responders, as in
                # kc_analysis? (maybe even put it in plot_traces?)
                # TODO maybe adapt whole mpl gui w/ updating db response calls
                # into here?

                # TODO could try to use cnmf ordering if i ever get that
                # working...

                if top_components:
                    odor_order_trace_gs = all_blocks_trace_gs[0, i]

                    # TODO probably move this inside plot_traces for re-use?
                    responsiveness = window_trial_means.groupby('cell').mean()
                    cellssorted = responsiveness.sort_values(ascending=False)
                    top_cells = cellssorted.index[:n]
                    pdf = comparison_df[comparison_df.cell.isin(top_cells)]
                    # TODO TODO TODO maybe also get a few top responders to each
                    # odor (w/ some visual division between odor groups)

                    # TODO maybe allow passing movie in to not have to load it
                    # multiple times when plotting traces on same data?
                    # (then just use self.movie)
                    viz.plot_traces(pdf, show_footprints=False,
                        gridspec=odor_order_trace_gs, n=n,
                        title='Top components'
                    )
                    presentation_order_trace_gs = all_blocks_trace_gs[1, i]
                    viz.plot_traces(pdf, show_footprints=False,
                        gridspec=presentation_order_trace_gs,
                        order_by='presentation_order', n=n
                    )

                if random_components:
                    if top_components:
                        orow = 2
                        prow = 3
                    else:
                        orow = 0
                        prow = 1

                    odor_order_trace_gs = all_blocks_trace_gs[orow, i]
                    viz.plot_traces(comparison_df, footprints=footprints,
                        gridspec=odor_order_trace_gs, n=n, random=True,
                        title='Random components'
                    )
                    presentation_order_trace_gs = all_blocks_trace_gs[prow, i]
                    viz.plot_traces(comparison_df, footprints=footprints,
                        gridspec=presentation_order_trace_gs,
                        order_by='presentation_order', n=n, random=True
                    )

            if plot_correlations:
                missing_dff = comparison_df[comparison_df.df_over_f.isnull()][
                    window_trial_means.index.names + ['df_over_f']
                ]
                # This + pivot_table w/ dropna=False won't work until this bug:
                # https://github.com/pandas-dev/pandas/issues/18030 is fixed.
                '''
                window_trial_means = pd.concat([window_trial_means,
                    missing_dff.set_index(window_trial_means.index.names
                    ).df_over_f
                ])
                '''

                print('plotting correlations...', end='', flush=True)
                # TODO rename to 'mean_df_over_f' or something, to avoid
                # confusion
                trial_by_cell_means = window_trial_means.to_frame().pivot_table(
                    index=['name1','name2','repeat_num','order'],
                    columns='cell', values='df_over_f'
                ).T

                # Hack to workaround pivot NaN behavior bug mentioned above.
                assert missing_dff.df_over_f.isnull().all()
                missing_dff.df_over_f = missing_dff.df_over_f.fillna(0)
                extra_cols = missing_dff.pivot_table(
                    index='cell', values='df_over_f',
                    columns=['name1','name2','repeat_num','order']
                )
                extra_cols.iloc[:] = np.nan

                assert (len(trial_by_cell_means.columns.drop_duplicates()) ==
                    len(trial_by_cell_means.columns)
                )
                trial_by_cell_means = pd.concat([trial_by_cell_means,
                    extra_cols], axis=1
                )
                assert (len(trial_by_cell_means.columns.drop_duplicates()) ==
                    len(trial_by_cell_means.columns)
                )
                # TODO modify to follow global-order-across-experiments that B
                # wanted + that i think i have an implementation of in
                # kc_mix_analysis.py
                trial_by_cell_means.sort_index(axis='columns', inplace=True)
                # end of the hack to workaround pivot NaN behavior


                trial_mean_presentation_order = \
                    trial_by_cell_means.sort_index(axis=1, level='order')

                corr_cbar_label = (r'Mean response $\frac{\Delta F}{F}$'
                    ' correlation'
                )

                odor_order_trial_mean_corrs = trial_by_cell_means.corr()

                if odor_order is not None:
                    odor_order_trial_mean_corrs = \
                        odor_order_trial_mean_corrs.reindex(odor_order,
                            level='name1', axis=0
                        )
                    odor_order_trial_mean_corrs = \
                        odor_order_trial_mean_corrs.reindex(odor_order,
                            level='name1', axis=1
                        )

                # TODO maybe just re-order odor_order_... corrs...
                presentation_order_trial_mean_corrs = \
                    trial_mean_presentation_order.corr()

                odor_order_ax = corr_axes[0, i]
                ticklabels = viz.matlabels(odor_order_trial_mean_corrs,
                    util.format_mixture
                )
                viz.matshow(odor_order_trial_mean_corrs,
                    ticklabels=ticklabels,
                    group_ticklabels=True,
                    cbar_label=corr_cbar_label,
                    ax=odor_order_ax,
                    fontsize=6
                )
                self.mpl_canvas.draw()

                presentation_order_ax = corr_axes[1, i]
                ticklabels = viz.matlabels(presentation_order_trial_mean_corrs,
                    util.format_mixture
                )
                viz.matshow(presentation_order_trial_mean_corrs,
                    ticklabels=ticklabels,
                    cbar_label=corr_cbar_label,
                    ax=presentation_order_ax,
                    fontsize=6
                )
                self.mpl_canvas.draw()

                print(' done', flush=True)

        if plot_odor_abbreviation_key:
            abbrev2odor = {v: k for k, v in odor2abbrev.items()}
            cell_text = []

            if odor_order is None:
                abbrev_iter = sorted(abbrev2odor.keys())
            else:
                abbrev_iter = odor_order

            # TODO TODO include concentration in odor column / new column
            print('\nOdor abbreviations:')
            for k in abbrev_iter:
                if k != 'paraffin' and k != NO_ODOR:
                    cell_text.append([k, abbrev2odor[k]])
                    print('{}: {}'.format(k, abbrev2odor[k]))
            print('')

            abbrev_key_ax.axis('off')
            abbrev_key_ax.table(cellText=cell_text, cellLoc='center',
                colLabels=['Abbreviation', 'Odor'], loc='center', bbox=None
            )

        # TODO maybe delete this...
        self.fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
        self.mpl_canvas.draw()
        # TODO maybe allow toggling same pane between avg and movie?
        # or separate pane for movie?
        # TODO use some non-movie version of pyqtgraph ImageView for avg,
        # to get intensity sliders? or other widget for that?

        self.processing = False

        self.display_params_editable(True)

        # TODO probably disable when running? or is it OK to upload stuff
        # during run? would any state variables have been overwritten?
        self.make_block_labelling_btns()
        self.block_label_btn_widget.setEnabled(True)

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
                '', 'JSON (*.json)', options=options
            )
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
            # TODO TODO why is this path not getting called when invoked from
            # editable param tab?? it should be
            # also says first arg (in branch below) is bool...
            self.params.to_json(self.default_json_params)
        elif len(args) == 1:
            json_str = args[0]
            self.save_json(json_str, self.default_json_params)
        else:
            raise ValueError('incorrect number of arguments')


    # TODO TODO add support for deleting presentations from db if reject
    # something that was just accepted?


    # TODO move core of this to util and just wrap it here
    # TODO fail earlier / handle in case where corresponding entry in "flies"
    # table is missing (like if number was not entered in fly_preps gsheet.
    # now it fails only on sql insert here, and message doesn't indicate how to
    # fix it.
    def upload_segmentation_info(self) -> None:
        run_info = {
            'run_at': [self.run_at],
            'recording_from': self.started_at,
            'input_filename': self.tiff_fname,
            #'input_md5': self.tiff_md5,
            'input_mtime': self.tiff_mtime,
            'start_frame': self.start_frame,
            'stop_frame': self.stop_frame,
            'parameters': self.parameter_json,
            'ijroi_file_path': self.ijroi_file_path,
            # TODO maybe share data / data browser similarly?
            'who': self.main_window.user,
            'host': socket.gethostname(),
            'host_user': getpass.getuser()
        }
        run = pd.DataFrame(run_info)
        run.set_index('run_at', inplace=True)

        # TODO depending on what table is in method callable, may need to make
        # pd index match sql pk?
        # TODO test that result is same w/ or w/o method in case where row did
        # not exist, and that read shows insert worked in w/ method case
        if self.ACTUALLY_UPLOAD:
            run.to_sql('analysis_runs', conn, if_exists='append',
                method=db.pg_upsert
            )

        # TODO worth preventing (attempts to) insert code versions and
        # pairings with analysis_runs, or is that premature optimization?

        # TODO nr_code_versions, rig_code_versions (both lists of dicts)
        # TODO ti_code_version (dict)
        if self.tiff_fname.endswith('_nr.tif'):
            mocorr_version_varname = 'nr_code_versions'
        elif self.tiff_fname.endswith('_rig.tif'):
            mocorr_version_varname = 'rig_code_versions'
        else:
            raise NotImplementedError

        # TODO TODO TODO if i'm gonna require this variable, check for it
        # in open_recording, so there are no surprises
        ti_code_version = matlab.get_matfile_var(self.matfile,
            'ti_code_version'
        )
        mocorr_code_versions = matlab.get_matfile_var(self.matfile,
            mocorr_version_varname, require=False
        )
        code_versions = [this_code_version]
        if self.parameter_json is not None:
            code_versions.append(caiman_code_version)
        else:
            assert self.ijroi_file_path is not None
        code_versions = (code_versions + ti_code_version +
            mocorr_code_versions
        )
        # TODO maybe impute missing mocorr version in some cases?

        if self.ACTUALLY_UPLOAD:
            db.upload_analysis_info(self.run_at, code_versions)

        # TODO TODO zoom to a constant factor (+ redraw) before saving, since
        # the zoom seems to change the way the figure looks?
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
        '''
        print('png_buff nbytes:', png_buff.getbuffer().nbytes)
        print('svg_buff nbytes:', svg_buff.getbuffer().nbytes)
        print('fig_buff nbytes:', fig_buff.getbuffer().nbytes)
        print('png_buff sizeof:', sys.getsizeof(png_buff))
        print('svg_buff sizeof:', sys.getsizeof(svg_buff))
        print('fig_buff sizeof:', sys.getsizeof(fig_buff))
        '''
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
        segrun_row['blocks_accepted'] = [None] * self.n_blocks
        # TODO TODO should i also be setting self.accepted and
        # self.to_be_accepted to the same thing?

        self.current_segrun_widget = self.add_segrun_widget(
            self.current_recording_widget, segrun_row)

        if self.ACTUALLY_UPLOAD:
            segmentation_run.to_sql('segmentation_runs', conn,
                if_exists='append', method=db.pg_upsert)

        # TODO unless i change db to have a canonical analysis version per
        # blocks/trials, probably want to check analysis version to be set
        # canonical has at least all the same blocks accepted as the previous
        # canonical (not really applicable below, but in the case when someone
        # is manually setting canonical)
        # TODO what to do with this?
        '''
        if (any_accepted and
            pd.isnull(self.current_recording_widget.data(0, Qt.UserRole))):
            self.make_canonical(self.current_segrun_widget)
        '''

        # TODO filter out footprints less than a certain # of pixels in cnmf?
        # (is 3 pixels really reasonable?)
        if self.ACTUALLY_UPLOAD:
            db.to_sql_with_duplicates(self.footprint_df, 'cells')

        self.uploaded_common_segrun_info = True


    def update_seg_accepted(self, segrun_treeitem, block_num, accepted) -> None:
        row = segrun_treeitem.data(0, Qt.UserRole)

        # TODO to get rid of some false positives here, could actually check
        # db for existance of presentations / traces for a given block
        # right now, intial accept -> relabel as rejected -> restart relabelling
        # -> try to accept, should fail, though it technically does not need to.
        if (self.relabeling_db_segrun and
            not self.uploaded_block_info[block_num]):

            # TODO maybe some option to force upload in meantime?
            #raise NotImplementedError('would need to re-run CNMF to get traces')
            pass

        row.blocks_accepted[block_num] = accepted
        run_at = pd.Timestamp(row.run_at)

        '''
        db_presentations = pd.read_sql_query(
            'SELECT presentation_id, comparison, presentation_accepted FROM' +
            " presentations WHERE analysis = '{}'".format(run_at), conn)
        print(db_presentations)
        '''

        # TODO maybe this weirdness is a sign i should just always be uploaded
        # (something like) the presentations
        # TODO maybe just make checks like this, and get rid of
        # uploaded_block_info stuff?
        if len(db_presentations) == 0 and accepted:
            raise ValueError('can not update presentations that are not in db')

        sql = ("UPDATE presentations SET presentation_accepted = " +
            "{} WHERE analysis = '{}' AND comparison = {}").format(
            accepted, run_at, block_num)
        ret = conn.execute(sql)
        '''
        print(pd.read_sql_query('SELECT comparison, presentation_accepted FROM'+
            " presentations WHERE analysis = '{}'".format(run_at), conn))
        '''

        # TODO maybe take out call to color_recording_node here if going to just
        # call once at end of upload anyway? flag to disable?
        self.color_segrun_node(segrun_treeitem)
        '''
        if accepted:
            # Starts out this way
            canonical = False
        else:
            canonical = None
        self.mark_canonical(segrun_treeitem, canonical)
        '''


    def upload_block_info(self, block_num):
        """Uploads traces and other metadata for block.
        """
        assert not self.relabeling_db_segrun

        if self.uploaded_block_info[block_num]:
            return

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
            presentation_dfs = self.presentation_dfs[
                (self.presentations_per_block * block_num):
                (self.presentations_per_block * (block_num + 1))
            ]
            comparison_dfs = self.comparison_dfs[
                (self.presentations_per_block * block_num):
                (self.presentations_per_block * (block_num + 1))
            ]

            for presentation_df, comparison_df in zip(
                presentation_dfs, comparison_dfs):

                db.to_sql_with_duplicates(presentation_df.drop(
                    columns='temp_presentation_id'), 'presentations')

                db_presentations = pd.read_sql('presentations', conn,
                    columns=(key_cols + ['presentation_id'])
                )

                presentation_ids = (db_presentations[key_cols] ==
                    presentation_df[key_cols].iloc[0]).all(axis=1)

                assert presentation_ids.sum() == 1, \
                    'presentation_id could not be determined uniquely'

                presentation_id = db_presentations.loc[presentation_ids,
                    'presentation_id'].iat[0]

                comparison_df['presentation_id'] = presentation_id

                # TODO TODO fix this bug (see first bug txt file on 5/20)
                # sqlalchemy.exc.ProgrammingError: (psycopg2.ProgrammingError)
                # table "temp_responses" does not exist
                # SQL: DROP TABLE temp_responses
                db.to_sql_with_duplicates(comparison_df.drop(
                    columns='temp_presentation_id'), 'responses'
                )

        self.uploaded_block_info[block_num] = True


    # TODO maybe make all block / upload buttons gray until current upload
    # finishes (mostly to avoid having to think about whether not doing so could
    # possibly cause a problem)?
    # TODO TODO TODO rename from upload_cnmf to be inclusive of ijroi case
    # TODO prompt to confirm whenever any operation (including exiting)
    # would not lose current traces before they have been uploaded
    def upload_cnmf(self) -> None:
        # TODO test
        if (any(self.to_be_accepted) and
            all([pd.isnull(x) for x in self.accepted])):
            # TODO might want to save something that doesn't include rejected
            # blocks? (or indicate in block which were?)

            fig_filename = (self.run_at.strftime('%Y%m%d_%H%M_') +
                self.recording_title.replace('/','_')
            )
            fig_path = join(analysis_output_root, 'figs')

            # TODO TODO save the initial traces (+ in future pixel corr plot) as
            # well, in at least this case, but maybe also just on load?
            # TODO but at that point, maybe just factor out the plotting and
            # generate in kc_natural_mixes/populate_db?
            # TODO TODO env var for fig output (might want in one flag
            # directory)
            # TODO make subdirs if they don't exist
            # TODO this consistent w/ stuff saved from toolbar in gui?
            png_path = join(fig_path, 'png', fig_filename + '.png')
            print('Saving fig to {}'.format(png_path))
            self.fig.savefig(png_path)

            svg_path = join(fig_path, 'svg', fig_filename + '.svg')
            print('Saving fig to {}'.format(svg_path))
            self.fig.savefig(svg_path)

            # TODO TODO also save to csv/flat binary/hdf5 per (date, fly,
            # thorimage) (probably at most, only when actually accepted.
            # that or explicit button for it.)

        if not self.uploaded_common_segrun_info:
            #
            print('uploading common segmentation info...', flush=True)
            #
            self.upload_segmentation_info()

        # TODO keep track of whether block specific stuff has been uploaded
        # (the stuff that only gets uploaded on accept)
        for i, (curr, target) in enumerate(zip(self.accepted,
            self.to_be_accepted)):
            #
            print('i: {}, already accepted: {}, target accepted: {}'.format(
                i, curr, target), flush=True
            )
            #

            if curr == target:
                continue

            if curr is not None:
                #
                print('updating_seg_accepted')
                #
                # TODO TODO TODO should not be able to set one to true if we
                # don't already have traces in db for it!!!
                self.update_seg_accepted(self.current_segrun_widget, i, target)

            elif target == True:
                #
                print('upload_block_info')
                #
                self.upload_block_info(i)

            self.accepted[i] = target
            # (all of the below comments copied from old accept_cnmf fn)
            # TODO just calculate metadata outright here?

            # TODO TODO save file to nas (particularly so that it can still be
            # there if database gets wiped out...) (should thus include
            # parameters [+version?] info)

            # TODO and refresh stuff in validation window s.t. this experiment
            # now shows up

            # TODO maybe also allow more direct passing of this data to other
            # tab

            # TODO TODO just save a bunch of different versions of the df/f,
            # computed w/ extract / detrend, and any key changes in arguments,
            # then load that and plot some stuff for troubleshooting

            # TODO move / delete. trying to restrict CNMF specific stuff to
            # run_cnmf now, so that other segmentation / trace extraction can be
            # used (like ijroi stuff)
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
            # TODO TODO maybe factor into fn and put behind hotkey / button
            # to save for inspection at will... (but should i fix cnmf
            # serialization first?)
            # TODO delete me
            # intended to use this to find best detrend / extract dff method
            '''
            # TODO to test extract_DF_F, need Yr, A, C, bl
            # detrend_df_f wants A, b, C, f (YrA=None, but maybe it's used?
            # in this fn, they call YrA the "residual signals")
            sliced_movie = self.cnm.get_sliced_movie(self.movie)
            Yr = self.cnm.get_Yr(sliced_movie)
            ests = self.cnm.estimates

            print('saving CNMF state to cnmf_state.p for debugging', flush=True)
            try:
                state = {
                    'Yr': Yr,
                    'A': ests.A,
                    'C': ests.C,
                    'bl': ests.bl,
                    'b': ests.b,
                    'f': ests.f,
                    'df_over_f': self.df_over_f,
                    # TODO raw_f too? or already included in one of these
                    # things?
                    'trial_start_frames': self.trial_start_frames,
                    'trial_stop_frames': self.trial_stop_frames,
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
            '''
            # TODO could also add some check / cleanup routines for orphaned
            # rows in presentations table (and maybe some other tables)
            # maybe share w/ code that checks distinct to decide whether to
            # load / analyze?
        self.color_segrun_node(self.current_segrun_widget)
        self.upload_btn.setEnabled(False)
        print('done uploading')


    def color_block_btn(self, block_num, accepted) -> None:
        if accepted:
            color = self.accepted_color
        else:
            color = self.rejected_color
        qss = 'background-color: {}'.format(color)
        self.block_label_btns[block_num].setStyleSheet(qss)


    def make_block_labelling_btns(self, *args) -> None:
        for btn in self.block_label_btns:
            self.block_label_btn_layout.removeWidget(btn)
            sip.delete(btn)

        self.block_label_btns = []

        have_accept_states = True
        any_mismatch = False
        if len(args) == 0:
            self.to_be_accepted = []
            self.accepted = []
            self.uploaded_block_info = []
            have_accept_states = False

        elif len(args) == 1:
            accepted = args[0]
            to_be_accepted = accepted

        elif len(args) == 2:
            accepted = args[0]
            to_be_accepted = args[1]
            for curr, target in zip(accepted, to_be_accepted):
                if curr != target:
                    any_mismatch = True
                    break

        else:
            raise ValueError('wrong number of arguments')

        enable_upload = any_mismatch
        if have_accept_states:
            if any([pd.isnull(x) for x in to_be_accepted]):
                enable_upload = False

        for i in range(self.n_blocks):
            block_btn = QPushButton(str(i + 1), self.block_label_btn_widget)
            block_btn.clicked.connect(partial(self.label_block, i))
            self.block_label_btn_layout.addWidget(block_btn)

            self.block_label_btns.append(block_btn)

            if have_accept_states:
                self.color_block_btn(i, to_be_accepted[i])
                # TODO TODO TODO also want to set any of
                # accepted/to_be_accepted/uploaded_block_info here?
            else:
                self.to_be_accepted.append(None)
                self.accepted.append(None)
                self.uploaded_block_info.append(False)

        self.block_label_btn_widget.setEnabled(have_accept_states)
        self.upload_btn.setEnabled(enable_upload)


    def label_block(self, block_num) -> None:
        curr_state = self.to_be_accepted[block_num]
        if curr_state is None or curr_state == False:
            self.to_be_accepted[block_num] = True
        elif curr_state == True:
            self.to_be_accepted[block_num] = False
        self.color_block_btn(block_num, self.to_be_accepted[block_num])

        # TODO factor into fn? i call this code one other place
        any_mismatch = False
        for curr, target in zip(self.accepted, self.to_be_accepted):
            if curr != target:
                any_mismatch = True
                break

        enable_upload = any_mismatch
        if any([pd.isnull(x) for x in self.to_be_accepted]):
            enable_upload = False

        self.upload_btn.setEnabled(enable_upload)
        #


    # TODO maybe support save / loading cnmf state w/ their save/load fns w/
    # buttons in the gui? (maybe to some invisible cache?)
    # (would need to fix cnmf save (and maybe load too) fn(s))


    # TODO replace self.sender().currentItem() w/ extra args as in expand
    # handling below?
    def handle_treeitem_dblclick(self):
        # TODO would this still work if enabling multi selection of top level
        # stuff, for hiding / whatever?
        # (like so.com/questions/6925011 )
        curr_item = self.sender().currentItem()
        if curr_item.parent() is None:
            self.current_recording_widget = curr_item
            if not (self.current_segrun_widget is not None and
                self.current_segrun_widget.parent() is curr_item):

                self.current_segrun_widget = None

            self.open_recording(curr_item)
        else:
            self.current_recording_widget = curr_item.parent()
            self.current_segrun_widget = curr_item
            self.open_segmentation_run(curr_item)


    def handle_treeitem_expand(self, recording_node):
        assert recording_node.parent() is None
        idx = self.data_tree.indexOfTopLevelItem(recording_node)
        tiff = self.motion_corrected_tifs[idx]

        seg_runs_loaded = recording_node.data(0, Qt.UserRole)
        if seg_runs_loaded:
            return
        tif_seg_runs = db.list_segmentations(tiff)
        recording_node.setData(0, Qt.UserRole, True)

        if tif_seg_runs is None:
            # TODO delete this if i find a cheap way to find whether there ARE
            # segmentation runs to be loaded, and use that to set the correct
            # child indicator initially
            recording_node.setChildIndicatorPolicy(
                QTreeWidgetItem.DontShowIndicatorWhenChildless)
            #
            return

        for _, r in tif_seg_runs.iterrows():
            self.add_segrun_widget(recording_node, r)

        self.color_recording_node(recording_node)


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


    # TODO possible to speed this up a little? maybe precompute things / don't
    # request as much data?
    # TODO TODO test case where this or open_recording are triggered
    # when cnmf is running / postprocessing
    # (what happens? what should happen? maybe just disable callbacks during?)
    def open_segmentation_run(self, segrun_widget):
        # TODO nothing should happen if trying to open same segrun that's
        # already open
        if not self.relabeling_db_segrun and self.movie is not None:
            self.previous_run_at = self.run_at
            self.previous_accepted = self.accepted
            self.previous_to_be_accepted = self.to_be_accepted
            self.previous_uploaded_common_segrun_info = \
                self.uploaded_common_segrun_info
            self.previous_uploaded_block_info = self.uploaded_block_info
            self.previous_n_blocks = self.n_blocks
            self.previous_footprint_df = self.footprint_df

        self.relabeling_db_segrun = True
        self.uploaded_common_segrun_info = True

        row = segrun_widget.data(0, Qt.UserRole)
        self.run_at = row.run_at
        # TODO rename to "ImageJ ROIs from" in that case
        # (how to test which case again? nullness of some fields?)
        print('\nCNMF run from {}'.format(util.format_timestamp(self.run_at)))

        # TODO for debugging. probably delete
        # TODO TODO also print whether analysis run was accepted or not
        db_presentations = pd.read_sql_query(
            'SELECT presentation_id, comparison, presentation_accepted FROM' +
            " presentations WHERE analysis = '{}'".format(
            pd.Timestamp(self.run_at)), conn
        )
        if len(db_presentations) > 0:
            print('Presentations in db for this segmentation run:')
            print(db_presentations)
        else:
            print('No presentations in db for this segmentation run.')
        print('')
        #

        # Since this is being non-None currently indicates self.footprint_df
        # coming from ImageJ ROIs in save_ijrois...
        self.ijroi_file_path = None
        self.footprint_df = pd.read_sql_query('''SELECT * FROM cells
            WHERE segmentation_run = '{}' '''.format(pd.Timestamp(self.run_at)),
            conn, index_col='cell'
        )
        assert len(self.footprint_df) > 0

        self.accepted = db.accepted_blocks(self.run_at)
        self.n_blocks = len(self.accepted)

        self.to_be_accepted = list(self.accepted)
        # Assumes that anything that was already accepted has had block info
        # uploaded. Could be false if I screwed something up before.
        self.uploaded_block_info = list(self.accepted)

        self.make_block_labelling_btns(self.accepted)

        # TODO hide param widget as appropriate in ijroi case
        # (probably a fn / pair of fns for this. probably want to do more than
        # just hide it)
        # TODO maybe make another widget on the stack to display
        # notification we are using ijroi stuff + info about it? or just
        # hide?
        if not pd.isnull(row.parameters):
            self.delete_other_param_widgets()
            self.param_display_widget = self.make_cnmf_param_widget(
                row.parameters, editable=False
            )
            self.param_widget_stack.addWidget(self.param_display_widget)
            self.param_widget_stack.setCurrentIndex(1)

        # TODO also load correct data params
        # is it a given that param json reflects correct data params????
        # if not, may need to model after open_recording

        # TODO if clear was taking a lot of time here, and can just make a new
        # one, why not do that for each time i plot? like for the trace, etc?
        # memory leak or something? mpl internals get screwed up?
        #print('plotting...', end='', flush=True)
        # TODO anything possible to make deserializing plots quicker?
        # (with complicated ones, can take ~5s)
        # maybe decrease # points in timeseries? could be excessive now
        t0 = time.time()
        self.fig = pickle.load(BytesIO(row.output_fig_mpl))
        self.mpl_canvas.figure = self.fig
        self.fig.canvas = self.mpl_canvas
        fig_w_inches, fig_h_inches = self.fig.get_size_inches()
        self.set_fig_size(fig_w_inches, fig_h_inches)
        # TODO test case where canvas was just drawing something larger
        # (is draw area not fully updated?)
        self.mpl_canvas.draw()
        #print(' done ({:.2f}s)'.format(time.time() - t0), flush=True)

        # TODO TODO should probably delete any traces in db if select reject
        # (or maybe just ignore and let other scripts clean orphaned stuff up
        # later? so that we don't have to regenerate traces if we change our
        # mind again and want to label as accepted...)


    # TODO maybe selecting two (/ multiple) analysis runs then right clicking
    # (or other way to invoke action?) and diffing parameters
    # TODO maybe also allow showing footprints overlayed or comparing somehow
    # (in same viewer?)


    def plot_avg_trace(self, recalc_trace=True):
        if recalc_trace:
            self.full_frame_avg_trace = util.full_frame_avg_trace(self.movie)

        # TODO TODO maybe make a new fig / do something other than clearing it
        # clearing seems to take a while sometimes (timing steps in
        # back-to-back open_seg... calls)
        self.fig.clear()
        # TODO maybe add some height for pixel based correlation matrix if i
        # include that
        fig_w_inches = 7
        fig_h_inches = 7
        self.set_fig_size(fig_w_inches, fig_h_inches)

        ax = self.fig.add_subplot(111)
        # TODO maybe replace w/ pyqtgraph video viewer roi, s.t. it can be
        # restricted to a different ROI if that would help

        # TODO TODO smooth, just don't have that cause artifacts at the edges
        '''
        smoothed_ff_avg_trace = util.smooth_1d(self.full_frame_avg_trace,
            window_len=7
        )
        ax.plot(smoothed_ff_avg_trace)
        '''
        ax.plot(self.full_frame_avg_trace)
        ax.set_title(self.recording_title)
        self.fig.tight_layout()
        self.mpl_canvas.draw()


    # TODO TODO fix how this fails when switching from output of a run of ijrois
    # (& perhaps also cnmf) analysis that was just generated and not uploaded.
    # looks like movie isn't reloaded?
    # (or actually maybe the movie just failed to load?) but if that was the
    # case, i could still try to run ijroi analysis, which should be prevented

    # TODO TODO TODO load last edited movie as soon as gui finishes loading
    # (or at least have a setting that can enable this)
    # TODO TODO TODO factor out the core of this fn to util
    def open_recording(self, recording_widget):
        # TODO TODO fix bug where block label btns stay labelled as in last
        # segrun that was open (seen after opening a recording but not analyzing
        # it, then switching back to that recording after opening a segrun)
        # (not an issue when switching to a different recording, it seems)

        # TODO maybe use setData and data instead?
        idx = self.data_tree.indexOfTopLevelItem(recording_widget)
        tiff = self.motion_corrected_tifs[idx]
        del idx
        # TODO TODO still want to switch view back to movie view in this case
        # (the avg trace and all). i think it's returning prematurely...
        if self.tiff_fname == tiff:
            if self.relabeling_db_segrun:
                self.accepted = self.previous_accepted
                self.to_be_accepted = self.previous_to_be_accepted
                self.uploaded_common_segrun_info = \
                    self.previous_uploaded_common_segrun_info
                self.uploaded_block_info = self.previous_uploaded_block_info
                self.run_at = self.previous_run_at
                self.n_blocks = self.previous_n_blocks
                self.footprint_df = self.previous_footprint_df

                self.plot_avg_trace(recalc_trace=False)

                # correct?
                self.make_block_labelling_btns(self.accepted,
                    self.to_be_accepted
                )

            self.relabeling_db_segrun = False
            return

        self.relabeling_db_segrun = False

        self.update_param_tab_index(self.cnmf_ctrl_widget.param_tabs)
        self.delete_other_param_widgets()

        # Trying all the operations that need to find files before setting any
        # instance variables, so that if those fail, we can stay on the current
        # data if we want (without having to reload it).
        # (now all operations that are likely to fail are inside
        # `load_recording`)
        print('')
        try:
            # load_recording is now going to do more than just what originally
            # happened in open_recording. I added a comment indicating the
            # portion in load_recording where what used to be in open_recording
            # ends, but I may want to refactor things in here after finishing
            # the new load_recording implementation
            raise NotImplementedError
            data = db.load_recording(tiff)

        # TODO be more specific about the types of errors expected / remove try
        # except
        except Exception as e:
            # TODO matter whether we raise / return here?
            #raise
            # TODO also have this print go to stderr
            print('Loading experiment failed with the following error:')
            traceback.print_exc()
            return

        # TODO delete
        '''
        tiff_dir, tiff_just_fname = split(tiff)
        analysis_dir = split(tiff_dir)[0]
        del tiff_dir
        full_date_dir, fly_dir = split(analysis_dir)
        del analysis_dir
        date_dir = split(full_date_dir)[-1]

        date = datetime.strptime(date_dir, '%Y-%m-%d')
        fly_num = int(fly_dir)
        thorimage_id = util.tiff_thorimage_id(tiff_just_fname)
        del tiff_just_fname

        recordings = df.loc[
            (df.date == date) &
            (df.fly_num == fly_num) &
            (df.thorimage_dir == thorimage_id)
        ]
        recording = recordings.iloc[0]
        if recording.project != 'natural_odors':
            warnings.warn('project type {} not supported. skipping.'.format(
                recording.project
            ))
            return
        '''
        # Now loaded in load_experiment, but left as a reminder of memory
        # considerations.
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
        ########################################################################
        # End stuff more likely to fail
        ########################################################################
        # Need to make sure we don't think the output of CNMF from other data is
        # associated with the new data we load.
        self.cnm = None

        self.recording_title = recording_title

        # TODO try to reduce the number of these variables needed, in favor of
        # storing more things in a smaller number of pandas objects
        self.date = date
        self.fly_num = fly_num
        self.thorimage_id = thorimage_id
        self.started_at = started_at
        self.n_blocks = n_blocks_from_gsheet
        self.presentations_per_block = presentations_per_block
        self.odor_ids = odor_ids
        self.frame_times = frame_times
        self.block_first_frames = block_first_frames
        self.block_last_frames = block_last_frames
        self.odor_onset_frames = odor_onset_frames
        self.odor_offset_frames = odor_offset_frames
        self.first_block = first_block
        self.last_block = last_block

        self.footprint_df = None

        self.trial_start_frames = trial_start_frames
        self.trial_stop_frames = trial_stop_frames

        self.matfile = mat

        # put behind a flag or something?
        '''
        start = time.time()
        # TODO maybe md5 array in memory, to not have to load twice?
        # (though for most formats it probably won't be the same... maybe none)
        # (would want to do it before slicing probably?)
        self.tiff_md5 = util.md5(tiff)
        end = time.time()
        print('Hashing TIFF took {:.3f} seconds'.format(end - start))
        '''
        # TODO factor out? (probably move to load_experiment...)
        self.tiff_mtime = datetime.fromtimestamp(getmtime(tiff))

        self.start_frame = drop_first_n_frames
        self.stop_frame = None

        # TODO generalize to 3d+t case?
        self.avg = np.mean(self.movie, axis=0)
        self.tiff_fname = tiff

        self.data_params = thor.cnmf_metadata_from_thor(tiff)

        # TODO try to get rid of these here? (s.t. only made at end of cnmf run)
        # was some error in loading either old run or data after loading old
        # run, as is
        self.make_block_labelling_btns()

        # TODO TODO fix 'fr'. it should be a float! (<- fixed, but...
        # more generally, if type change in CNMF params, that type change
        # should be reflect indep of default param type, right?) (?)
        # TODO maybe in the meantime, err if trying to set float to int field?
        for i in range(self.data_group_layout.count()):
            # TODO TODO does it always alternate qlabel / widget for that label?
            # would that ever break? otherwise, how to get a row, with the label
            # and the other widget for that row?
            item = self.data_group_layout.itemAt(i).widget()
            if type(item) == QLabel:
                continue

            label = self.data_group_layout.labelForField(item).text()
            if label not in self.data_params:
                continue

            v = self.data_params[label]

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

        # TODO what was this for again? delete...
        '''
        # TODO just try to modify cnmf to work w/ uint16 as input?
        # might make it a bit faster...
        from_type = np.iinfo(self.movie.dtype)
        to_type = np.iinfo(np.dtype('float32'))

        to_type.max * (self.movie / from_type.max)
        '''

        self.plot_avg_trace()

        # TODO maybe allow playing movie somewhere in display widget?
        self.run_cnmf_btn.setEnabled(True)

        # TODO change to list / delete
        #self.uploaded_presentations = False
        self.uploaded_common_segrun_info = False


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
        # TODO TODO maybe bold current item, to be clear as to which is
        # currently selected, even if focus is moved in tree widget?
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

        tiff = util.motion_corrected_tiff_filename(*comp[:-1])
        # TODO move loading to some other process? QRunnable? progress bar?
        # TODO if not just going to load just comparison, keep movie loaded if
        # clicking other comparisons / until run out of memory
        # TODO just load part of movie for this comparison
        print('Loading TIFF {}...'.format(tiff), end='', flush=True)
        self.movie = tifffile.imread(tiff)
        print(' done')

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
        fps = thor.fps_from_thor(self.metadata)
        self.fps = fps
        self.response_frames = int(np.ceil(fps * response_calling_s))
        self.background_frames = int(np.floor(fps * np.abs(
            self.metadata.from_onset.apply(lambda x: x.min()).min())))

        self.full_footprints = dict()
        for cell, data in footprint_rows[['cell','x_coords','y_coords','weights'
            ]].set_index('cell').iterrows():
            x_coords, y_coords, weights = data

            # TODO do i have some util fn that can accomplish this now?
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
        cropped_footprint, ((x_min, x_max), (y_min, y_max)) = crop_to_nonzero(
            footprint, margin=6
        )

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
                    near_footprint_contours[c] = viz.plot_closed_contours(f, ax,
                        colors='blue'
                    )

                contour = viz.plot_closed_contours(cropped_footprint, ax,
                    colors='red'
                )
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
        # TODO make focus clear when clicked out of / esc (editingFinished?)
        user_label = QLabel('User', self.user_widget)
        # TODO maybe attribute on people as to whether to show here or not?
        user_select = QComboBox(self.user_widget)

        # TODO something like a <select user> default?
        self.user_cache_file = '.cnmf_gui_user.p'
        self.user = self.load_default_user()

        self.nicknames = set(pd.read_sql('people', conn)['nickname'])
        if self.user is not None:
            pd.DataFrame({'nickname': [self.user]}).set_index('nickname'
                ).to_sql('people', conn, if_exists='append',
                method=db.pg_upsert)

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
        print('initializing segmentation tab...', flush=True) #, end='')
        self.seg_tab = Segmentation(self)
        #print(' done')
        print('done')
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
        tools = menutil.addMenu('&Tools')
        tools.addAction(debug_action)
        '''
        debug_shortcut = QShortcut(QKeySequence('Ctrl+d'), self)
        debug_shortcut.activated.connect(self.debug_shell)

        self.setFocus()


    def change_user(self, user):
        if user not in self.nicknames:
            pd.DataFrame({'nickname': [user]}).to_sql('people', conn,
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
        # TODO cause problems? alt? (does at least seem to work if there are cv2
        # windows)
        cv2.destroyAllWindows()


    def debug_shell(self):
        import ipdb
        ipdb.set_trace()


def main():
    global presentations
    global recordings
    global recordings_meta
    global footprints
    global comp_cols
    global caiman_code_version
    global this_code_version

    # Calling this first to minimize chances of code diverging.
    # TODO might be better to do before some imports... some of the imports are
    # kinda slow
    # TODO will __file__ still work if i get to the point of installing this
    # package w/ pip?
    caiman_code_version, this_code_version = [util.version_info(m,
        used_for='extracting footprints and traces')
        for m in [caiman, __file__]
    ]

    # TODO TODO matlab stuff that only generated saved output needs to be
    # handled, and it can't work this way.

    # TODO TODO TODO maybe use gitpython to check for remote updates and pull
    # them / relaunch / fail until user pulls them?

    # TODO TODO also deal w/ matlab code versionS somehow...

    # TODO maybe rename all of these w/ db_ prefix or something, to
    # differentiate from non-global versions in segmentation tab code
    print('reading odors from postgres...', end='', flush=True)
    odors = pd.read_sql('odors', conn)
    print(' done')

    print('reading presentations from postgres...', end='', flush=True)
    presentations = pd.read_sql('presentations', conn)
    print(' done')

    presentations['from_onset'] = presentations.from_onset.apply(
        lambda x: np.array(x))

    presentations = db.merge_odors(presentations, odors)

    # TODO change sql for recordings table to use thorimage dir + date + fly
    # as index?
    recordings = pd.read_sql('recordings', conn)

    presentations = db.merge_recordings(presentations, recordings)

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

    footprints = pd.read_sql('cells', conn)
    footprints = footprints.merge(recordings_meta, on='recording_from')
    footprints.set_index(util.recording_cols, inplace=True)

    comp_cols = util.recording_cols + ['comparison']

    print('starting MATLAB engine...', end='', flush=True)
    matlab.matlab_engine()
    print(' done')

    # TODO convention re: this vs setWindowTitle? latter not available if making
    # a window out of a widget?
    # TODO maybe setWindowTitle based on the tab? or add to end?
    app = QApplication(['2p analysis GUI'])
    win = MainWindow()
    win.showMaximized()
    app.exit(app.exec_())


if __name__ == '__main__':
    main()

