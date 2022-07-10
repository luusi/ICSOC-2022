# ICSOC 2022 submission

Implementation of <insert title of paper>

## Preliminaries

We assume the review uses a UNIX-like machine and that has Python 3.8 installed.

- Set up the virtual environment. 
First, install [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/).
Then:
```
pipenv install --dev
```
                    
- this command is to start a shell within the Python virtual environment (to be done whenever a new terminal is opened):
```
pipenv shell
```

- Install the Python package in development mode:
```
pip install -e .
# alternatively:
# python setup.py develop 
```

- To use rendering functionalities, you will also need to install Graphviz. 
  At [this page](https://www.graphviz.org/download/) you will
  find the releases for all the supported platform.

- Install [Lydia](https://github.com/whitemech/lydia). 
  We suggest to [use the Docker installation](https://github.com/whitemech/lydia#use-the-docker-image).

## How to run the code

- Run a Jupyter Notebook server:

```
jupyter-notebook
```

- Open the link and navigate through `docs/notebooks` and run the notebook `01-electric-motor-production.ipynb` to
  replicate the experiment.

## License

The software is released under the MIT license.
