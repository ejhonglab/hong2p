#!/usr/bin/env python3

"""
GUI to load movies, cell boundaries, and to store ratings on the quality of the
boundaries in a database.
"""

# TODO factor away need for this
import socket
from os.path import split

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
     QVBoxLayout, QListWidget)
import pyqtgraph as pg
import tifffile
import numpy as np
import pandas as pd
# TODO factor away need for this
from sqlalchemy import create_engine


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

        # TODO TODO one top widget for (matplotlib based?) trial dffs
        self.annotation_layout.addWidget(self.imv)

        # TODO TODO bottom widget for labelling


    def open_comparison(self):
        idx = self.sender().currentRow()
        comp = tuple(self.fly_comps.iloc[idx])
        metadata = self.presentations.loc[comp]

        cells = self.footprints.cell.loc[comp[:-1]]
        self.cell_list.clear()
        self.cell_list.addItems([str(c) for c in sorted(cells.to_list())])

        # TODO TODO load (all / part of) movie


    def annotate_cell_comp(self):
        # TODO do something else if not just in-order only
        idx = self.sender().currentRow()

        # TODO TODO compute / plot trial dFFs w/ contour
        # (just embed my other mpl code?)

        # TODO TODO crop movie to just around footprint
        # overlay the footprint + neighboring ones in diff color
        # (hopefully possible in pyqtgraph...)
        

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

