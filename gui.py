#!/usr/bin/env python3

"""
GUI to do make ROIs for trace extraction and validate those ROIs.
"""

# TODO factor away need for this
import socket
from os.path import split, join, exists
import xml.etree.ElementTree as etree

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, 
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QPushButton)
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


db_hostname = 'atlas'
our_hostname = socket.gethostname()
if our_hostname == db_hostname:
    url = 'postgresql+psycopg2://tracedb:tracedb@localhost:5432/tracedb'
else:
    url = ('postgresql+psycopg2://tracedb:tracedb@{}' +
        ':5432/tracedb').format(db_hostname)

conn = create_engine(url)

# TODO [refactor to (?)] make it possible to refresh these?
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


class MotionCorrection(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Motion Correction')


class Segmentation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Segmentation / Trace Extraction')


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

        #self.comp_movie = self.movie[self.first_frame:self.last_frame, :, :]
        print('movie.shape:', self.movie.shape)
        #print('comp_movie.shape:', self.comp_movie.shape)

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
            footprint = np.array(coo_matrix((weights, (x_coords, y_coords)),
                shape=(256, 256)).todense())

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

        # TODO maybe some percentile / fixed size about maximum
        # density?
        # TODO TODO take the margin as a parameter in the GUI
        cropped_footprint, ((x_min, x_max), (y_min, y_max)) = \
            crop_to_nonzero(footprint, margin=6)

        near_footprints = dict()
        # TODO TODO TODO *find* nearby footprints->contours of another color
        # TODO TODO make sense to convert all footprints to full matrices in
        # open_comparison, to make searching easier?
        for c, f in self.full_footprints.items():
            if c == cell_id:
                continue

            # TODO maybe it should overlap by at least a certain number of
            # pixels? (i.e. pull back bounding box a little)
            in_display_region = f[x_min:x_max+1, y_min:y_max+1]
            if (in_display_region > 0).any():
                near_footprints[c] = in_display_region

        # TODO TODO how to draw only partial contours? ok to just draw out of
        # bounds?

        # TODO should probably compute over nearby footprints too...
        # (or just once somewhere / use known value of what min could be?
        # just all nonzero?)
        contour_level = cropped_footprint[cropped_footprint > 0].min()

        #######################################################################

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
                near_footprint_contours = dict()
                for c, f in near_footprints.items():
                    mpl_near_contour = ax.contour(f, [contour_level],
                        colors='blue')
                    assert len(mpl_near_contour.collections) == 1
                    paths = mpl_near_contour.collections[0].get_paths()
                    assert len(paths) == 1
                    near_footprint_contours[c] = paths[0].vertices

                # TODO factor into fn
                mpl_contour = ax.contour(cropped_footprint, [contour_level],
                    colors='red')
                assert len(mpl_contour.collections) == 1
                paths = mpl_contour.collections[0].get_paths()
                assert len(paths) == 1
                contour = paths[0].vertices
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

        # TODO TODO TODO are these necessary? (ROIs seem to display w/o...
        # but will trace data be wrong otherwise?)
        # TODO should i explicitly disconnect previous roi before deleting, or
        # does it not matter / is that not a thing?
        # TODO maybe just do this once at beginning w/ some kind of empty roi?
        pg_roi.sigRegionChanged.connect(self.imv.roiChanged)
        # TODO TODO compare intermediate results (stepping into pg code)
        # w/ PolyLineROI vs RectROI

        # TODO TODO TODO fix! options:
        # - change ImageView.roiChanged to not call w/ returnMappedCoords=True
        # - fix PolyLineROI s.t. it works w/ returnMappedCoords=True
        #   (currently, no classes besides parent ROI seem to...)

        ##import ipdb; ipdb.set_trace()
        self.imv.roiChanged()

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
        self.tabs.addTab(self.seg_tab, self.seg_tab.windowTitle())

        val_index = self.tabs.addTab(self.validation_tab,
            self.validation_tab.windowTitle())

        self.tabs.setCurrentIndex(val_index)


def main():
    # TODO convention re: this vs setWindowTitle? latter not available if making
    # a window out of a widget?
    # TODO maybe setWindowTitle based on the tab? or add to end?
    app = QApplication(['2p analysis GUI'])
    win = MainWindow()
    win.showMaximized()
    app.exit(app.exec_())

if __name__ == '__main__':
    main()

