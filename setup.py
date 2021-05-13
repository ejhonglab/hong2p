
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

        'PyQt5',
        'pyqtgraph',

        'matplotlib',
        'seaborn',

        'scikit-learn',
        'gitpython',
        'ipdb',

        # TODO move cv2 in here (opencv-python)?
        # (only gui.py/extract_template.py and some other test scripts import
        # it on module import, but maybe still nice to have by default?)
    ],
    tests_require=['pytest'],
    entry_points={
        # I define all of these in hong2p/cli_entry_points.py, and then modify
        # hong2p/__init__.py to import each of them explicitly.
        'console_scripts': [
            'thor2tiff=hong2p:thor2tiff_cli',
            'showsync=hong2p:showsync_cli',
            'suite2p-params=hong2p:suite2p_params_cli',
        ],
    },
    author="Tom O'Connell",
    author_email='toconnel@caltech.edu',
    license='GPLv3',
    keywords='caiman cnmf imaging neuroscience olfaction drosophila',
    url='https://github.com/ejhonglab/python_2p_analysis',
    description=('GUI and utils for olfaction calcium imaging, '
        'partially wrapping CaImAn'
    ),
    #long_description=open('README.rst').read(),
)
