FROM python:3.14

RUN pip install --no-cache-dir notebook jupyterlab

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER}

RUN adduser --disabled-password --gecos "Default user" --uid ${NB_UID} ${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_USER} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

RUN pip install --no-cache-dir ".[interactive]"
