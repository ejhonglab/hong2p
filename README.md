
### Installation

Activate a virtual environment if you'd like, then:

```
pip install git+https://github.com/ejhonglab/hong2p
```

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

In RStudio, install the `reticulate` package via Tools -> Install Packages.

Then copy paste the code in the `Example usage` section below into a new script in R
studio.

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

# This does not require separate installation of hong2p. `pip install -e .` would also
# work if you don't care about tests.
pip install -e .[test]
```

#### Testing

```
# If for some reason `pytest --version` prints a path seemingly not managed by your
# virtual environment (where it says "...imported from <path>"), you may benefit from
# using `python -m pytest` instead.
pytest
```
