#!/usr/bin/env python3

"""
GUI to load movies, cell boundaries, and to store ratings on the quality of the
boundaries in a database.
"""

# TODO factor away need for this
import socket
from os.path import split, join, exists
import xml.etree.ElementTree as etree

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
     QVBoxLayout, QListWidget, QPushButton)
import pyqtgraph as pg
import tifffile
import numpy as np
import pandas as pd
# TODO factor away need for this
from sqlalchemy import create_engine
# TODO factor all of these out as much as possible
from scipy.sparse import coo_matrix
import cv2
###import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


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


recording_cols = [
    'prep_date',
    'fly_num',
    'thorimage_id'
]
# TODO use env var like kc_analysis currently does for prefix after refactoring
analysis_output_root = '/mnt/nas/mb_team/analysis_output'


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #######################################################################
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
        self.presentations = presentations.set_index(comp_cols)
        self.footprints = footprints

        # TODO TODO option to operate at recording level instead

        self.fly_comps = \
            presentations[comp_cols].drop_duplicates().sort_values(comp_cols)

        # TODO TODO and how to map back to the right index when clicked?
        items = ['/'.join(r.split()[1:])
            for r in str(self.fly_comps).split('\n')[1:]]

        self.list = QListWidget(self)
        self.list.addItems(items)
        self.list.itemDoubleClicked.connect(self.open_comparison)

        # TODO or table w/ clickable rows? existing code for that?
        # TODO TODO could use QTableView
        # (see mfitzp/15-minute-apps currency example)

        # TODO where clicking an element starts evaluation process in that data

        # TODO maybe rename comparison according to mixture

        #######################################################################

        # TODO TODO maybe just make this the average image and overlay clickable
        # rois / make cells clickable?
        self.cell_list = QListWidget(self)
        self.cell_list.itemDoubleClicked.connect(self.annotate_cell_comp)

        # TODO TODO maybe (pop up?) some kind of cell selector that lets you
        # pick a single cell to focus on? w/ diff display for / restriction to
        # those that have not yet been annotated?

        # TODO TODO TODO for now, maybe forgo play of whole video / roi
        # selection in there, and just go through random order of cells
        # showing trial dffs and (downsampled) movie

        # TODO worth storing a reference to it like that? examples seem to do
        # this, but maybe they actually wanted to call methods on the thing?
        self.imv = pg.ImageView()

        # TODO TODO TODO how to actually use play fn / other fns that let you
        # change frame number w/o having to explicitly find that frame and
        # setData? how does that work? setData seems to only take single frames?

        # TODO does setImage need to happen after show? show generally just
        # called on mainwindow / top widget, and recurse down?
        self.imv.setImage(np.zeros((256, 256)))
        '''
        self.movie = np.random.rand(50, 256, 256)
        self.imv.setImage(self.movie)
        fps = 12
        # TODO how to make it loop?
        self.imv.play(fps)
        '''

        # TODO make image take up full space and have scroll go through
        # timesteps? or at least have scroll over time bar do that?
        # TODO empty display area until something selected?
        # or only display list until something is selected?

        # TODO should this have some explicit parent?
        self.central_widget = QWidget()
        # TODO can't just use mainwindow thing? (seemed to not work...)
        # just subclass qwidget then?
        self.setCentralWidget(self.central_widget)

        # TODO TODO labels above things in hboxlayout
        # TODO want to use addStretch?
        self.layout = QHBoxLayout(self.central_widget)
        self.central_widget.setLayout(self.layout)
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

        # TODO need to specify figsize?
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

        # TODO should i have a separate view / horz concatenated movies to show
        # each trial side-by-side, or should i have one big movie?
        self.annotation_layout.addWidget(self.imv)

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

    
    def open_comparison(self):
        idx = self.sender().currentRow()
        comp = tuple(self.fly_comps.iloc[idx])
        self.fly_comp = comp

        # TODO rewrite to avoid performancewarning (indexing past lexsort depth)
        cells = self.footprints.cell.loc[comp[:-1]]
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

        #self.comp_movie = self.movie[self.first_frame:self.last_frame, :, :]
        print('movie.shape:', self.movie.shape)
        #print('comp_movie.shape:', self.comp_movie.shape)

        # TODO TODO could allow just comparison to be played... might be useful
        # (until a cell is clicked)

        # TODO maybe put the odors for this comparison in some text display
        # somewhere?

        # TODO maybe make response_calling_s a parameter modifiable in the GUI
        response_calling_s = 3.0
        # TODO TODO TODO get fps from thor (check this works...)
        fps = fps_from_thor(self.metadata)
        self.fps = fps
        self.response_frames = int(np.ceil(fps * response_calling_s))
        self.background_frames = int(np.floor(fps * np.abs(
            self.metadata.from_onset.apply(lambda x: x.min()).min())))


    def annotate_cell_comp(self):
        # TODO do something else if not just in-order only
        # TODO TODO just use text converted to int...
        cell_id = self.sender().currentRow()

        (prep_date, fly_num, thorimage_id, comparison_num) = self.fly_comp
        #######################################################################
        # TODO probably worth also just showing the average F (as was used,
        # exclusively, before)?
        '''
        avg = cv2.normalize(avg, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        better_constrast = cv2.equalizeHist(avg)
        '''
        cell_row = (prep_date, fly_num, thorimage_id, cell_id)

        # TODO just store the same data w/ two diff indexes to save time?
        footprints = self.footprints.reset_index().set_index(
            recording_cols + ['cell'])
        footprint_row = footprints.loc[cell_row]

        # TODO TODO TODO *find* nearby footprints->contours of another color

        weights, x_coords, y_coords = \
            footprint_row[['weights','x_coords','y_coords']]

        # TODO TODO get this shape from thor metadata
        footprint = np.array(coo_matrix((weights,
            (x_coords, y_coords)), shape=(256, 256)).todense())

        # TODO maybe some percentile / fixed size about maximum
        # density?
        cropped_footprint, ((x_min, x_max), (y_min, y_max)) = \
            crop_to_nonzero(footprint, margin=6)
        contour_level = cropped_footprint[cropped_footprint > 0].min()

        # TODO TODO TODO clear the figure/all axes as necessary
        mpl_contour = None
        for ax, row in zip(self.axes.flat,
            self.metadata.sort_values('odor_onset_frame').itertuples()):

            odor_onset_frame = row.odor_onset_frame
            odor_offset_frame = row.odor_offset_frame
            curr_start_frame = odor_onset_frame - self.background_frames
            curr_stop_frame = odor_onset_frame + self.response_frames

            # TODO TODO delete comp_movie (not useful because frame #s
            # change...)
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

            if mpl_contour is None:
                mpl_contour = ax.contour(cropped_footprint, [contour_level],
                    colors='red')
                assert len(mpl_contour.collections) == 1
                paths = mpl_contour.collections[0].get_paths()
                assert len(paths) == 1
                contour = paths[0].vertices
            else:
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

        # TODO does clearing this get rid of other default things i want?
        # (yes, it seems so...)
        # TODO need to keep track of ROI items?
        # TODO TODO is just deleting the polylinerois sufficient to return the
        # viewbox close enough to its original state...? not clear

        print(view_box.allChildren())
        print(view_box.allChildItems())
        print(len(view_box.allChildren()))
        print(len(view_box.allChildItems()))

        to_remove = []
        for c in view_box.allChildren():
            if type(c) == pg.graphicsItems.ROI.PolyLineROI:
                to_remove.append(c)
        for r in to_remove:
            view_box.removeItem(r)

        print(view_box.allChildren())
        print(view_box.allChildItems())
        print(len(view_box.allChildren()))
        print(len(view_box.allChildItems()))

        #import ipdb; ipdb.set_trace()

        # TODO isn't there some util fn to convert from numpy? use that?
        # TODO need to specify closed? doing what i expect?
        # TODO show red
        pg_roi = pg.PolyLineROI(contour, closed=True)
        # TODO TODO need to call view_box.invertY(True) as in RectROI
        # programcreek example ("upper left origin")? X?
        # so far, seems though inversion is not necessary
        # (assuming movie is not flipped wrt matplotlib images)
        # TODO TODO but movie may be... (seemingly along both axes?)

        # TODO TODO are bounds right? (some rois are getting clipped now)
        # TODO TODO what about this is making the video not display anymore?!?
        self.imv.getView().addItem(pg_roi)

        # TODO TODO how to make it loop?
        self.imv.play(self.fps)
        # TODO TODO TODO how to overlay contour??

        # TODO TODO convert contour to pg.PolyLineROI?
        # or does that not really make drawing the ROI any easier?
        

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

        # TODO combine w/ current cell/comparison info and...
        # TODO TODO insert values into database
        

    # TODO TODO TODO hotkeys / buttons for labelling

    # TODO TODO some kind of tab system for switching between loading recordings
    # / doing cnmf on that vs. loading comparisons to evaluate w/ more
    # granularity? or keep granularity the same and change the evaluation or
    # CNMF?

    # TODO TODO if doing cnmf in this gui, save output directly to database


def main():
    app = QApplication(['Segmentation Validation GUI'])
    win = MainWindow()
    win.showMaximized()
    app.exit(app.exec_())

if __name__ == '__main__':
    main()

