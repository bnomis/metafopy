# Pinned R environment for regenerating tests/reference fixtures.
#
#   docker build -t metafopy-r tests/r
#   docker run --rm -v "$PWD":/work -w /work metafopy-r Rscript tests/r/generate_fixtures.R
FROM rocker/r-ver:4.5.1

ARG METAFOR_VERSION=5.2-1
ARG METADAT_VERSION=1.6-0

RUN Rscript -e 'install.packages(c("remotes", "jsonlite"), repos = "https://cloud.r-project.org")' \
 && Rscript -e 'remotes::install_version("metadat", version = Sys.getenv("METADAT_VERSION"), repos = "https://cloud.r-project.org", upgrade = "never")' \
 && Rscript -e 'remotes::install_version("metafor", version = Sys.getenv("METAFOR_VERSION"), repos = "https://cloud.r-project.org", upgrade = "never")'
