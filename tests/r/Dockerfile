# Pinned R environment for regenerating tests/reference fixtures.
#
#   docker build -t metafopy-r tests/r
#   docker run --rm -v "$PWD":/work -w /work metafopy-r Rscript tests/r/generate_fixtures.R
#   docker run --rm -v "$PWD":/work -w /work metafopy-r Rscript tests/r/generate_escalc.R
FROM rocker/r-ver:4.5.1

ARG METAFOR_VERSION=5.2-1
ARG METADAT_VERSION=1.6-0

# gsl (UCOR, SMCR with vtype="UB") and mvtnorm (RTET) are needed by tests/r/generate_escalc.R
RUN apt-get update && apt-get install -y --no-install-recommends libgsl-dev && rm -rf /var/lib/apt/lists/*
RUN Rscript -e 'install.packages(c("remotes", "jsonlite", "gsl", "mvtnorm", "CompQuadForm"), repos = "https://cloud.r-project.org")' \
 && Rscript -e 'remotes::install_version("metadat", version = Sys.getenv("METADAT_VERSION"), repos = "https://cloud.r-project.org", upgrade = "never")' \
 && Rscript -e 'remotes::install_version("metafor", version = Sys.getenv("METAFOR_VERSION"), repos = "https://cloud.r-project.org", upgrade = "never")'
