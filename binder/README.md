# Binder

`Dockerfile` builds the image for the Binder badge of `docs/notebooks/showcase.ipynb`, see <https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html>.
It starts from the official Python image because metafopy needs Python 3.14.7 or later and repo2docker's conda base image has 3.14.0 (mamba cannot upgrade it in place).
