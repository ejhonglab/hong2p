### Building the documentation

Sphinx configuration is in `source/conf.py`. API documentation `.rst` files are
generated under `source/apidoc/` (gitignored) by `sphinxcontrib.apidoc` on each build,
so they should not be edited manually. To exclude a module, add it to
`apidoc_excluded_paths` in `source/conf.py`.


#### Setup

From the root of the repository (requires Python >= 3.12):
```
pip install -e .[doc]
```

The documentation dependencies are pinned in the `doc` extra in `../pyproject.toml`.


#### Build and publish

From the root of the repository:
```
hong2p-docs
git push
```

`hong2p-docs` (defined in `hong2p/cli_entry_points.py:build_docs_cli`):
1. Deletes any previous `docsrc/build` and `docsrc/source/apidoc` outputs.
2. Builds the HTML (equivalent to `make html` here).
3. Replaces `../docs` with the contents of `build/html`. This includes the empty
   `.nojekyll` file Github Pages requires, which `sphinx.ext.githubpages` generates.
4. Commits only `../docs` (refusing to run if other changes are staged).

Pass `--no-commit` to skip step 4. Github Pages should be configured to serve the
`docs/` directory of the `master` branch, which publishes it
[here](https://ejhonglab.github.io/hong2p).


#### Manual method

To just build the HTML without touching `../docs`:
```
make html
```

Then you can open `build/html/index.html` in a web browser to view the output.
