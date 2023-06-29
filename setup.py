
from setuptools import setup, find_packages

setup(
    name='hong2p',
    version='0.0.0',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    install_requires=[
        # This one might not be strictly necessary. The errs that seemed to
        # happen during pip install w/o it might not be fatal, assuming other
        # installation options exist besides the .whl format (like sdists).
        'wheel',

        # Note: this caused natural_odors/requirements/literature_data.txt
        # install to fail for Sharne because she didn't have postgres installed.
        #'psycopg2',
        #'sqlalchemy',
        'h5py',
        'numpy',
        'scipy',
        'tifffile',

        'pandas',
        'xarray',

        'PyQt5',
        'pyqtgraph',

        'matplotlib',
        'seaborn',

        'scikit-learn',
        'gitpython',

        'PyYAML',

        'jinja2',
        # How does what PyLaTeX offers compare? As of 2023, PyLaTeX last updated 2021,
        # while [mbr/]latex was last updated 2019.
        'latex',

        # https://stackoverflow.com/questions/32688688
        # TODO move all imports of this to top-level, since i don't think it's heavy
        # enough to warrant conditional imports
        'ijroi @ git+https://github.com/tom-f-oconnell/ijroi',

        'platformdirs',

        'ipdb',

        # TODO move cv2 in here (opencv-python)?
        # (only gui.py/extract_template.py and some other test scripts import
        # it on module import, but maybe still nice to have by default?)
    ],
    extras_require={
        'test': [
            'pytest',

            # Provides `cv2`. Used in a few particular functions and imported early in
            # the definitions of those functions. Tests of any of those functions would
            # need it installed though.
            'opencv-python',
        ],
    },
    entry_points={
        # I define all of these in hong2p/cli_entry_points.py, and then modify
        # hong2p/__init__.py to import each of them explicitly.
        'console_scripts': [
            'thor2tiff=hong2p:thor2tiff_cli',
            'showsync=hong2p:showsync_cli',
            'suite2p-params=hong2p:suite2p_params_cli',

            # Mainly to support shell functions that cd to these directories, e.g.
            # 2p() {
            #     cd "$(hong2p-data)"
            # }
            'hong2p-data=hong2p:print_data_root',
            'hong2p-raw=hong2p:print_raw_data_root',
            'hong2p-analysis=hong2p:print_analysis_intermediates_root',

            'thor-pairs=hong2p:print_paired_thor_subdirs',
            'thor-notes=hong2p:print_thorimage_subdir_notes',
        ],
    },
    author="Tom O'Connell",
    author_email='toconnel@caltech.edu',
    license='GPLv3',
    keywords='imaging neuroscience olfaction drosophila',
    url='https://github.com/ejhonglab/hong2p',
    description=('GUI and utils for olfaction calcium imaging, '
        'partially wrapping CaImAn'
    ),
    #long_description=open('README.rst').read(),
)
