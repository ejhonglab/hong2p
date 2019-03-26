#!/usr/bin/env python3

"""
GUI to do make ROIs for trace extraction and validate those ROIs.
"""

import socket
import sys
from os.path import split, join, exists, sep
import xml.etree.ElementTree as etree
import warnings
from collections import defaultdict
import traceback
from functools import partial
import glob

# TODO first three are in QtCore for sure, but the rest? .Qt? .QtGUI?
# differences?
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThreadPool, QRunnable
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QListWidget,
    QGroupBox, QPushButton, QLineEdit, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QLabel)

import pyqtgraph as pg
import tifffile
import numpy as np
import pandas as pd
# TODO factor away need for this
from sqlalchemy import create_engine
# TODO factor all of these out as much as possible
from scipy.sparse import coo_matrix
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
# TODO TODO find uncommitted changes + git version immediately upon run
# allow reloading caiman code w/o closing gui?
import git

# TODO TODO allow things to fail gracefully if we don't have cnmf.
# either don't display tabs that involve it or display a message indicating it
# was not found.
from caiman.source_extraction.cnmf import params, cnmf


# TODO also move these two variable defs? (to globals below?) for sake of
# consistency?
recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
# TODO use env var like kc_analysis currently does for prefix after refactoring
raw_data_root = '/mnt/nas/mb_team/raw_data'
analysis_output_root = '/mnt/nas/mb_team/analysis_output'


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


def list_motion_corrected_tifs():
    """
    """
    motion_corrected_tifs = []
    for date_dir in glob.glob(join(analysis_output_root, '**')):
        for fly_dir in glob.glob(join(date_dir, '**')):
            try:
                fly_num = int(split(fly_dir)[-1])

                tif_dir = join(fly_dir, 'tif_stacks')
                if exists(tif_dir):
                    # TODO explicitly filter for *_nr.tif / *_rig.tif?
                    motion_corrected_tifs += glob.glob(join(tif_dir, '*.tif'))

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Motion Correction')


# TODO either make a superclass for everything that should share a data browser,
# or just have one data browser that interacts with other widgets (tabwidget may
# have to not contain data browser in that case?)
# one argument for trying to have each tab contain a data browser is that then
# each widget (which is currently a tab) definition could more readily by run on
# it's own, without having to combine with a data browser
class Segmentation(QWidget):
    def __init__(self):
        super().__init__()
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

        items = []
        for d in self.motion_corrected_tifs:
            x = d.split(sep)
            fname_parts = x[-1].split('_')
            cor_type = 'rigid' if fname_parts[-1][:-4] == 'rig' else 'non-rigid'
            item_parts = [x[-4], x[-3], '_'.join(fname_parts[:-1]), cor_type]
            items.append('/'.join(item_parts))
            # TODO make make tooltip the full path? or show full path somewhere
            # else?

        # TODO maybe make slider between data viewer and everything else?
        # sliders for everything in this layout?
        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        self.list = QListWidget(self)
        self.layout.addWidget(self.list)
        self.list.setFixedWidth(210)
        self.list.addItems(items)
        self.list.itemDoubleClicked.connect(self.open_recording)

        # Other groups are: motion, online, preprocess_params
        # TODO worth including patch_params? preprocess_params?
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
        # TODO TODO get rid of most of this (maybe whole data tab if decay_time
        # moved to temporal) after loading from thorlabs metadata
        dont_show_by_group.update({
            'data': {'fnames','dims'}, #'fr','dxy'},
            'merging': {'gSig_range'},
        })
            
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

        param_dict = self.params.to_dict()

        n = int(np.ceil(np.sqrt(len(cnmf_groups))))
        seen_types = set()
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

            print(g)
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

                if print_stuff:
                    print(k, v, type(v))
                    print(doc_line)
                    print('')

                group_layout.addRow(k, w)
            print('')

        print('Seen types:', seen_types)

        # TODO set tab index to most likely to change? spatial?

        # TODO TODO is there really any point to displaying anything here?
        # almost seems like button->dialog or something would be more
        # appropriate if not...
        # does the division make sense b/c some validation takes longer than
        # others (and that is what can be in the separate validation tab)?

        self.display_widget = QWidget()
        self.layout.addWidget(self.display_widget)
        self.display_layout = QVBoxLayout(self.display_widget)
        self.display_widget.setLayout(self.display_layout)
        # TODO TODO add toolbar / make image navigable

        self.fig = Figure()
        self.mpl_canvas = FigureCanvas(self.fig)
        self.display_layout.addWidget(self.mpl_canvas)
        # TODO TODO accept / reject dialog beneath this
        # (maybe move save button out of ctrl widget? move reject in there?)
        # TODO TODO rejection should mark set of params in database
        # TODO warn if would run analysis on same data w/ same params as had
        # previously led to a rejection
        # TODO TODO accept should save file to nas and load to database
        # TODO and refresh stuff in validation window s.t. this experiment now
        # shows up
        # TODO maybe also allow more direct passing of this data to other tab

        self.contour_ax = self.fig.subplots(1, 1)
        self.contour_ax.axis('off')

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
        # TODO separate button to cancel? change run-button to cancel?

        self.contour_ax.clear()
        self.contour_ax.axis('off')

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

        # Execute
        self.threadpool.start(worker)


    def run_cnmf(self):
        # TODO time cnmf run + report how long it took
        print('running cnmf')
        # TODO TODO use cm.cluster.setup_cluster or not? required now?
        # TODO what is the dview that returns as second arg?
        # TODO i feel like this should be written to not require n_processes...
        n_processes = 1
        # TODO need to copy params first?
        # TODO TODO maybe make CNMF obj w/ params immediately, as it seems to
        # change some of the parameters? then modify params from there?
        self.cnm = cnmf.CNMF(n_processes, params=self.params)
        # images : mapped np.ndarray of shape (t,x,y[,z])
        # TODO does it really need to be mapped? how can you even check if a
        # numpy array is mapped?
        # TODO maybe use fit_file for certain ways of getting here in the gui?
        # TODO TODO TODO check dims / C/F order
        self.cnm.fit(self.movie)

        # TODO maybe have a widget that shows the text output from cnmf?


    def cnmf_done(self):
        self.run_cnmf_btn.setEnabled(True)
        print('done with cnmf')


    def get_cnmf_output(self):
        # TODO TODO allow toggling between type of backgrouond image shown
        # (radio / combobox for avg, snr, etc? "local correlations"?)
        self.cnm.estimates.plot_contours(img=self.avg, ax=self.contour_ax,
            display_numbers=False, colors='r', linewidth=1.0)
        self.mpl_canvas.draw()
        ######import ipdb; ipdb.set_trace()


    def save_default_params(self):
        print('Writing new default parameters to {}'.format(
            self.default_json_params))
        self.params.to_json(self.default_json_params)
        # TODO maybe use pickle for this?

    # TODO maybe support save / loading cnmf state w/ their save/load fns w/
    # buttons in the gui? (maybe to some invisible cache?)


    def open_recording(self):
        # TODO TODO avoid circularity problem (right now, only stuff w/ CNMF
        # output is put in database), so either:
        # - put non-cnmf data in db w/o cnmf stuff, asap, and also load that
        #   stuff here
        # - or detect stuff not in db, put in browser / load for cnmf here, and
        #   then put in db when done
        idx = self.sender().currentRow()
        '''
        rec_row = tuple(self.recordings.iloc[idx])

        # TODO do i need this?
        self.metadata = self.presentations.loc[rec_row]

        tiff = motion_corrected_tiff_filename(*rec_row)
        '''
        tiff = self.motion_corrected_tifs[idx]

        data_params = cnmf_metadata_from_thor(tiff)
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

        # TODO why does this not seem to print until load finishes...? end?
        print('loading tiff {}...'.format(tiff), end='')
        # TODO is cnmf expecting float to be in range [0,1], like skimage?
        self.movie = tifffile.imread(tiff).astype('float32')
        print(' done.')

        self.avg = np.mean(self.movie, axis=0)
        '''
        # TODO just try to modify cnmf to work w/ uint16 as input?
        # might make it a bit faster...
        from_type = np.iinfo(self.movie.dtype)
        to_type = np.iinfo(np.dtype('float32'))

        to_type.max * (self.movie / from_type.max)
        '''
        # TODO maybe allow playing movie somewhere in display widget?


class ROIAnnotation(QWidget):
    def __init__(self):
        super().__init__()

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

    # TODO TODO some kind of tab system for switching between loading recordings
    # / doing cnmf on that vs. loading comparisons to evaluate w/ more
    # granularity? or keep granularity the same and change the evaluation or
    # CNMF?

    # TODO TODO if doing cnmf in this gui, save output directly to database


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # TODO maybe an earlier tab for motion correction or something at some
        # point?
        self.mc_tab = MotionCorrection()
        self.seg_tab = Segmentation()
        self.validation_tab = ROIAnnotation()

        # TODO factor add + windowTitle bit to fn?
        self.tabs.addTab(self.mc_tab, self.mc_tab.windowTitle())
        seg_index = self.tabs.addTab(self.seg_tab, self.seg_tab.windowTitle())

        val_index = self.tabs.addTab(self.validation_tab,
            self.validation_tab.windowTitle())

        #self.tabs.setCurrentIndex(val_index)
        self.tabs.setCurrentIndex(seg_index)


def main():
    global conn
    global odors
    global recordings
    global presentations
    global rec_meta_cols
    global recordings_meta
    global footprints
    global comp_cols

    db_hostname = 'atlas'
    our_hostname = socket.gethostname()
    if our_hostname == db_hostname:
        url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
    else:
        url = ('postgresql+psycopg2://tracedb:tracedb@{}' +
            ':5432/tracedb').format(db_hostname)

    conn = create_engine(url)

    print('reading odors from postgres...', end='')
    odors = pd.read_sql('odors', conn)
    print(' done')

    print('reading presentations from postgres...', end='')
    presentations = pd.read_sql('presentations', conn)
    print(' done')

    presentations['from_onset'] = presentations.from_onset.apply(
        lambda x: np.array(x))

    presentations = merge_odors(presentations, odors)

    # TODO change sql for recordings table to use thorimage dir + date + fly
    # as index?
    recordings = pd.read_sql('recordings', conn)

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

    footprints = pd.read_sql('cells', conn)
    footprints = footprints.merge(recordings_meta, on='recording_from')
    footprints.set_index(recording_cols, inplace=True)

    comp_cols = recording_cols + ['comparison']


    # TODO convention re: this vs setWindowTitle? latter not available if making
    # a window out of a widget?
    # TODO maybe setWindowTitle based on the tab? or add to end?
    app = QApplication(['2p analysis GUI'])
    win = MainWindow()
    win.showMaximized()
    app.exit(app.exec_())

if __name__ == '__main__':
    main()

