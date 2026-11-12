
### Documentation

Documentation [is available
here](https://ejhonglab.github.io/hong2p/apidoc/hong2p.html).


### Installation

Activate a virtual environment if you'd like, then:

```
pip install 'hong2p[all] @ git+https://github.com/ejhonglab/hong2p'
```

Without `[all]` (`pip install git+https://github.com/ejhonglab/hong2p`), only a few
lightweight dependencies are installed. That is enough for `hong2p.thor` (ThorImage /
ThorSync outputs), `hong2p.olf`, `hong2p.pandas`, and `hong2p.util`, but not for
`hong2p.viz`, `hong2p.io`, `hong2p.roi`, `hong2p.suite2p`, `hong2p.xarray`, or
`hong2p.latex` (importing these will raise an error saying to install `hong2p[all]`).
`hong2p.rdm` (odor-pair matrices: pair Series, distances between odors' responses),
`hong2p.stats` (residuals, correlations and resampling of grouped matrices) and
`hong2p.imaging` (response windows of traces, the indicator kernel, noise-corrected
distances) also import with the core dependencies, but their statistics and fits use
scipy / scikit-learn (`[all]`).

Projects depending on hong2p should include `[all]` in their requirement, unless they
only use the modules above, e.g. `hong2p[all] @ git+https://github.com/ejhonglab/hong2p`.


### Usage in R

#### Installation

First install Anaconda Python, then:
```
conda create --name r-reticulate python=3.6
conda activate r-reticulate

# If you computer doesn't have git installed, you will need to install that before this.
# The lab Linux computers should already have git installed.
pip install git+https://github.com/ejhonglab/hong2p
```

If you also use ROS on this computer, you will probably want to run:
```
conda config --set auto_activate_base false
```
...which I think is sufficient to prevent `conda` from interfering with ROS operation.

In RStudio, install the `reticulate` package via Tools -> Install Packages.

Then copy paste the code in the `Example usage` section below into a new script in R
studio.

#### Updating

```
conda activate r-reticulate
pip uninstall -y hong2p
pip install git+https://github.com/ejhonglab/hong2p
```

#### Example usage

Source (rather than Run) the following script in RStudio:
```
library(reticulate)

# Note that this is the name of the conda environment we created earlier.
use_condaenv("r-reticulate", required = TRUE)

thor <- import("hong2p.thor")

# This data should be available on the NAS (perhaps through FileZilla), under:
# Kristina/Imaging_data/Imaging_data_2021
thorimage_dir <- "/path/to/20210508_a/1_water"
thorsync_dir <- "/path/to/20210508_a/SyncData006"

# bounding_frames is a list of (first_frame, first_frame_after_odor, last_frame) indices
# suitable for indexing the corresponding ThorImage movie
bounding_frames <- thor$assign_frames_to_odor_presentations(thorsync_dir, thorimage_dir)
```

And you can interact with the `bounding_frames` variable in the R command line after it
finishes.

The directories referenced in `thorsync_dir` and `thorimage_dir` above must exist in
order for this example to work. Please change these variables to point to your local
copies of the same data, or you may also try changing these paths to point to other
outputs of ThorSync / ThorImage.


### Reporting a bug

If you encounter an error / unexpected result, please do these two things:

1. Upload any data necessary to reproduce the issue to the NAS under the
   `hong2p_test_data` directory.

2. Log in to Github and submit an issue [here](https://github.com/ejhonglab/hong2p/issues/new).
   Give your issue an appropriate title and in the comment field, please:
   - Paste any code necessary to reproduce your issue. Ideally, format as a code block.
   - If you uploaded data to the NAS under `hong2p_test_data`, also include the full
     paths to any relevant files / directories you uploaded.


### Development

```
cd <directory where you want your source code>
git clone https://github.com/ejhonglab/hong2p
cd hong2p
```

Activate a virtual environment if you'd like, then:

```
# Just since some of the default Python versions in Ubuntu have ancient pip versions
pip install --upgrade pip

# This does not require separate installation of hong2p. The `dev` extra includes
# `all`, plus test dependencies. `pip install -e .[all]` would also work if you don't
# care about tests.
pip install -e .[dev]
```

#### Updating the documentation

The documentation is generated via Sphinx (configured under `docsrc/`), and hosted on
Github Pages (serving the `docs/` directory on `master`). To re-generate the
documentation, from the root of this repository:
```
# Only needed once (requires Python >= 3.12). Installs hong2p editable, with the
# documentation dependencies (the `doc` extra in pyproject.toml).
pip install -e .[doc]

# Rebuilds docs/ from docsrc/ and commits the docs/ changes (only). Refuses to run if
# other changes are staged.
hong2p-docs

git push
```

There may be a slight delay between pushing and Github running the action to update the
Github Pages site.

To rebuild `docs/` without committing, use `hong2p-docs --no-commit`. To just generate
the docs to `docsrc/build/html` (without touching `docs/`), run `make html` from
`docsrc/` and open `docsrc/build/html/index.html`. See `docsrc/README.md` for more
details.


#### Testing

Tests require the `dev` extra (`pip install -e .[dev]`, as above).

To just run tests not marked as "slow":
```
pytest -m "not slow"
```

To run all tests:
```
pytest
```

To run tests under both the oldest supported versions (python 3.8, pandas 1.5) and the
current `./venv`:
```
tox
```

To check the core modules work with only the core dependencies (without `[all]`):
```
tox -e core
```
