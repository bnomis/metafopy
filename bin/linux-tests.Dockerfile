# The Linux of CI's runners (ubuntu-26.04), so glibc's libm matches. Both tags are exact; see bin/linux-tests.sh.
FROM ghcr.io/astral-sh/uv:0.13.8 AS uv

FROM ubuntu:26.04
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libarchive-tools \
    && rm -rf /var/lib/apt/lists/*
# bsdtar extracts the tree: GNU tar cannot create nested directories in an emulated (amd64 on arm64) container.
WORKDIR /work
