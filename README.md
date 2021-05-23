
### Installation

Activate a virtual environment if you'd like, then:

```
pip install git+https://github.com/ejhonglab/hong2p
```


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

# This does not require separate installation of # hong2p. `pip install -e .` would also
# work if you don't care about tests.
pip install -e .[test]
```

#### Testing

```
# If for some reason `which pytest` gives you a path seemingly not managed by your
# virtual environment, you may benefit from using `python -m pytest` instead.
pytest
```
