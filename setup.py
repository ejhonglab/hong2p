
from setuptools import setup

setup(
    name='hong2p',
    version='0.0.0',
    packages=['hong2p'],
    # TODO .py suffix here? populate_db too. subdir of hong_2p?
    #scripts=['gui'],
    setup_requires=['pytest-runner'],
    install_requires=['numpy', 'pandas'],
    tests_require=['pytest'],
    author="Tom O'Connell",
    author_email='toconnel@caltech.edu',
    license='GPLv3',
    keywords='caiman cnmf imaging neuroscience olfaction drosophila',
    url='https://github.com/ejhonglab/python_2p_analysis',
    description='GUI and utils for olfaction calcium imaging, ' +
        'partially wrapping CaImAn'#,
    #long_description=open('README.rst').read(),
)
