
#### Building the documentation

```
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

make html
```

Then you can open `build/html/index.html` in a web browser to view the output.
