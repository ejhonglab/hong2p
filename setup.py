
from setuptools import setup, find_packages

setup(
    name='hong2p',
    version='0.0.0',
    packages=find_packages(),
    # TODO .py suffix here? populate_db too. subdir of hong_2p?
    #scripts=['gui'],
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

        # 0.25.1 seemed to work for much of at least early 2020. More versions
        # also likely work, but if transferring pickled pandas objects between
        # installations, it becomes more important to match the pandas versions.
        # this version fails to build on 18.04 with python3.8 though
        #'pandas==0.25.1',
        'pandas',

        # Aiming to guarantee that at least the matplotlib backend 'Qt5Agg' will
        # work.  (but does system Qt5 need to be installed separately? order of
        # that vs. pip install matter?)
        # Clearing some pip cache might have also solved it, but I just picked
        # this specific version because
        # https://stackoverflow.com/questions/59768179 indicated it could fix an
        # error I encountered when installing version selected (5.14.2) without
        # specifying.
        #'PyQt5==5.14',
        'PyQt5',

        'matplotlib',
        'seaborn',

        'scikit-learn',
        'gitpython',
        'ipdb',

        'latex',
        'jinja2',
    ],
    #    # TODO some way to specify these urls in an authentication agnostic way,
    #    # so it defaults to ssh if we have that set up?
    #    # (might matter if i point to private repos this way)
    #    # TODO test these don't get installed to overwrite my local editable
    #    # copies, as long as i have the editable versions installed
    #    # TODO some way to only need to type in username if pip would actually
    #    # have to install these (now, if they are installed, pip still wants to
    #    # clone them)

    #    # If you do NOT have SSH authentication setup for your git, you will
    #    # need to change all of these URLs to the versions using the https://...
    #    # URLs. (assuming some of these are private, at least)

    #    '-e ../chemutils',
    #    #'git+ssh://git@github.com/ejhonglab/chemutils',
    #    #'git+https://github.com/tom-f-oconnell/drosolf',

    #    '-e ../drosolf',
    #    #'git+ssh://git@github.com/tom-f-oconnell/drosolf',
    #    #'git+https://github.com/ejhonglab/chemutils',

    #    # If using Matt Bauer's mushroom body model
    #    # seems I can't actually install a working copy this way
    #    # TODO double check this though, cause it looks like i have (maybe in
    #    # addition?) memory issues (probably not...)
    #    #'-e ../olfsysm',
    #    'git+ssh://git@github.com/ejhonglab/olfsysm',
    #    #'git+https://github.com/ejhonglab/olfsysm',

    #    # If using neuprint to parameterize Matt's model.
    #    '-e ../neuprint_helper',
    #    #'git+ssh://git@github.com/ejhonglab/neuprint_helper',
    #    #'git+https://github.com/ejhonglab/neuprint_helper',

    #    # TODO could specify my patched version of pyqtgraph w/ github link?
    #    # (neither of my main computers (atlas & blackbox) had my fork of
    #    # pyqtgraph in ~/src 2020-05-26, so what my fork was doing is not that
    #    # critical now. it still may be for some of the really old gui.py stuff,
    #    # specifically the first work i did making a gui to annotate the quality
    #    # of ROIs
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'thor2tiff=hong2p:thor2tiff',
        ],
    },
    author="Tom O'Connell",
    author_email='toconnel@caltech.edu',
    license='GPLv3',
    keywords='caiman cnmf imaging neuroscience olfaction drosophila',
    url='https://github.com/ejhonglab/python_2p_analysis',
    description='GUI and utils for olfaction calcium imaging, ' +
        'partially wrapping CaImAn'#,
    #long_description=open('README.rst').read(),
)
