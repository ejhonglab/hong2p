
### Building the documentation


#### Initial setup

```
# had been using 3.8.12
python3 -m venv venv

source venv/bin/activate

# had been using 20.2.3
pip install --upgrade pip

pip install -r requirements.txt
```

If `hong2p` requirements change (currently specified in `hong2p/setup.py`), 
re-run `pip install -r requirements.txt`. Otherwise, you may notice import errors as
part of the build process.


#### Build

```
make github
```

Should generate output under `docsrc/build`, and copy to `../docs`. Also makes and
copies `.nojekyll` file that was required for Github pages to work correctly with this
type of output (`make html` below didn't seem to include this file).

To publish, continue at step 4 under publishing section below (committing `../docs`
contents).


##### Manual method

Only relevant if `make github` is broken for some reason.

```
make html
```

Then you can open `build/html/index.html` in a web browser to view the output.


#### Publishing

1. Delete any existing `hong2p/docs` directory, via `git rm -rf docs` (if exists, all
   contents should have been generated from a prior run of the build process above).

2. Make `hong2p/docs` directory via `mkdir docs`.

3. Copy contents of `hong2p/docsrc/build/html` to `hong2p/docs`, via:
   `cp -a docsrc/build/html/* ./docs/.`

4. (if used `make github`, start here) Commit all contents to the repository.

5. Push changes to `ejhonglab/hong2p`, and the Github repository should be set up to
   publish that HTML [here](https://ejhonglab.github.io/hong2p).

