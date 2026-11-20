# CA_2026_FP dev environment -- SystemC + this project's conda env
# (`ca_fp_2026`, see environment.yml) only. Deliberately does NOT contain
# this repo's own source: this project is a private repo distributed as a
# ZIP, not `git clone`-able, so it's mounted as a volume at container run
# time instead of COPYed into the image (also keeps this image reusable
# across repo updates without a rebuild). See top-level README.md for the
# full environment-setup story this image is standing in for; the
# Troubleshooting table there still applies if something inside the
# container itself misbehaves.
#
# ── Pinned to the reference (grading) machine ───────────────────────────
# Every version below is pinned to what the instructor's machine actually
# builds and runs the simulator with, so the host compiler never matters:
# a student whose own system has e.g. GCC 16 still compiles with the GCC
# 13.3.0 inside this image. Nothing here floats to "latest":
#   - base image:  ubuntu:24.04 pinned by digest (amd64, created 2026-10-25)
#   - apt:         every package comes from the Ubuntu snapshot archive
#                  frozen at APT_SNAPSHOT, and the toolchain packages are
#                  additionally pinned to exact versions (see the list)
#   - SystemC:     3.0.0 source tarball, verified by sha256
#   - Miniforge:   fixed release, verified by sha256
#   - Python:      exact versions in environment.yml (CPU-only torch)
# A final RUN step checks the toolchain versions and fails the build if any
# of them differs, so a drifted image can't be built silently.
#
# Network access is needed once, at `docker build` time, to download the
# base image, Ubuntu packages, SystemC's public source release and the
# conda/pip packages -- nothing here ever touches this project's own
# (private) repository.
#
# Build (from this repo's root, so environment.yml is in the build context):
#   docker build -t ca_fp_env .
#
# Run (mounts your extracted repo at /workspace):
#   docker run -it --rm -v "$(pwd):/workspace" ca_fp_env
#   # inside the container (see USER_GUIDE.md for the full flow):
#   cd new_code && mkdir -p build && cd build && cmake ..

# x86_64 only (the pinned packages / Miniforge installer are amd64); Apple
# Silicon hosts run it under emulation.
FROM --platform=linux/amd64 ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

ARG DEBIAN_FRONTEND=noninteractive
# Ubuntu snapshot archive timestamp (https://snapshot.ubuntu.com). All apt
# packages resolve against the archive exactly as it was at this moment.
ARG APT_SNAPSHOT=20261101T000000Z
# SystemC release tag (accellera-official/systemc, IEEE 1666-2023 SystemC
# 3.0.0) -- matches the version this project has actually been built/run
# against; bump deliberately, not accidentally, if a newer release is
# needed later.
ARG SYSTEMC_VERSION=3.0.0
ARG SYSTEMC_SHA256=4d0ab814719cfd6b1d195dd4bcb1b9e6edc5881b9a3e44117336a691992bf779
ARG MINIFORGE_VERSION=26.9.4-1
ARG MINIFORGE_SHA256=571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc

# ── Switch apt to the snapshot archive ───────────────────────────────────
# snapshot.ubuntu.com is HTTPS-only and the base image has no CA bundle, so
# ca-certificates is installed first from the regular archive. It is pinned
# to the 24.04 release-pocket version, which never changes. It only affects
# TLS, not the compiled result.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates=20240203 \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?#URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/#" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && rm -rf /var/lib/apt/lists/*

# ── OS build toolchain, pinned to the reference machine's exact versions ─
# (versions from `dpkg-query -W` on the reference machine, 2026-11-03).
# binutils is pinned explicitly because the snapshot's noble-updates has a
# newer one (2.42-4ubuntu2.10) than the reference machine (.8).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-13=13.3.0-6ubuntu2~24.04.1 \
        g++-13=13.3.0-6ubuntu2~24.04.1 \
        cpp-13=13.3.0-6ubuntu2~24.04.1 \
        gcc-13-base=13.3.0-6ubuntu2~24.04.1 \
        gcc-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        g++-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        cpp-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        libgcc-13-dev=13.3.0-6ubuntu2~24.04.1 \
        libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1 \
        libstdc++6=14.2.0-4ubuntu2~24.04.1 \
        libgcc-s1=14.2.0-4ubuntu2~24.04.1 \
        gcc=4:13.2.0-7ubuntu1 \
        g++=4:13.2.0-7ubuntu1 \
        cpp=4:13.2.0-7ubuntu1 \
        libc6=2.39-0ubuntu8.9 \
        libc6-dev=2.39-0ubuntu8.9 \
        libc-dev-bin=2.39-0ubuntu8.9 \
        linux-libc-dev=6.8.0-147.147 \
        binutils=2.42-4ubuntu2.8 \
        binutils-common=2.42-4ubuntu2.8 \
        binutils-x86-64-linux-gnu=2.42-4ubuntu2.8 \
        libbinutils=2.42-4ubuntu2.8 \
        libctf0=2.42-4ubuntu2.8 \
        libctf-nobfd0=2.42-4ubuntu2.8 \
        libgprofng0=2.42-4ubuntu2.8 \
        libsframe1=2.42-4ubuntu2.8 \
        make=4.3-4.1build2 \
        dpkg-dev=1.22.6ubuntu6.6 \
        build-essential=12.10ubuntu1 \
        cmake=3.28.3-1build7 \
        cmake-data=3.28.3-1build7 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── SystemC, built from source against THIS image's own glibc/libstdc++ ──
# (not copied from any prebuilt install -- that would only be safe if the
# base image's glibc/libstdc++ ABI happened to match whatever machine built
# it; building fresh here removes that assumption entirely). Source comes
# from SystemC's own public GitHub release tarball, unrelated to this
# project's private repo. Same flags as the reference machine's install
# (Release, C++17).
ENV SYSTEMC_HOME=/opt/systemc
RUN wget -q "https://github.com/accellera-official/systemc/archive/refs/tags/${SYSTEMC_VERSION}.tar.gz" \
        -O /tmp/systemc.tar.gz \
    && echo "${SYSTEMC_SHA256}  /tmp/systemc.tar.gz" | sha256sum -c - \
    && mkdir -p /tmp/systemc-src \
    && tar -xzf /tmp/systemc.tar.gz -C /tmp/systemc-src --strip-components=1 \
    && cmake -S /tmp/systemc-src -B /tmp/systemc-build \
        -DCMAKE_INSTALL_PREFIX="${SYSTEMC_HOME}" \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/systemc-build -j"$(nproc)" \
    && cmake --install /tmp/systemc-build \
    && rm -rf /tmp/systemc.tar.gz /tmp/systemc-src /tmp/systemc-build

# ── Conda (Miniforge) + this project's own environment ────────────────────
ENV CONDA_HOME=/opt/conda
RUN wget -q "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        -O /tmp/miniforge.sh \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p "${CONDA_HOME}" \
    && rm /tmp/miniforge.sh

# environment.yml is read from THIS repo's build context (the directory
# you run `docker build` from) -- a single small spec file, not this
# project's actual source, and not fetched from anywhere. Keep this in
# sync with the top-level environment.yml by construction: this COPY
# always reads whatever that file currently says, nothing is duplicated
# by hand here.
COPY environment.yml /tmp/environment.yml
RUN "${CONDA_HOME}/bin/conda" env create -f /tmp/environment.yml \
    && "${CONDA_HOME}/bin/conda" clean -afy \
    && rm /tmp/environment.yml

# Put the ca_fp_2026 env's own bin/ ahead of everything else on PATH --
# this is what actually fixes the "cmake picks up the wrong python3"
# gotcha from the top-level README's Troubleshooting table: inside this
# container there is no "wrong" python3 to accidentally pick up, this one
# is simply always first. Applies to every process the container runs
# (interactive shell or a direct `docker run ... <cmd>`), not just
# `bash`-read shells -- deliberately not relying on `conda activate`
# needing to run first.
ENV PATH="${CONDA_HOME}/envs/ca_fp_2026/bin:${CONDA_HOME}/bin:${PATH}"

# ── Self-check: fail the build if anything drifted from the reference ────
# The versions are also written to /etc/ca_fp_env_versions.txt, so
# `cat /etc/ca_fp_env_versions.txt` inside a container shows what it has.
RUN set -e; \
    test "$(g++ -dumpfullversion)" = "13.3.0"; \
    test "$(dpkg-query -W -f='${Version}' g++-13)" = "13.3.0-6ubuntu2~24.04.1"; \
    test "$(dpkg-query -W -f='${Version}' libc6)" = "2.39-0ubuntu8.9"; \
    test "$(dpkg-query -W -f='${Version}' binutils)" = "2.42-4ubuntu2.8"; \
    test "$(cmake --version | head -1)" = "cmake version 3.28.3"; \
    grep -q '#define SC_VERSION_MAJOR *3' "${SYSTEMC_HOME}/include/sysc/kernel/sc_ver.h"; \
    python3 -c "import sys, numpy, yaml, torch; \
assert sys.version.split()[0] == '3.12.14', sys.version; \
assert numpy.__version__ == '2.5.0', numpy.__version__; \
assert yaml.__version__ == '6.0.3', yaml.__version__; \
assert torch.__version__ == '2.14.0+cpu', torch.__version__"; \
    { echo "g++      $(g++ -dumpfullversion)"; \
      echo "glibc    $(dpkg-query -W -f='${Version}' libc6)"; \
      echo "binutils $(dpkg-query -W -f='${Version}' binutils)"; \
      echo "cmake    $(cmake --version | head -1 | cut -d' ' -f3)"; \
      echo "SystemC  ${SYSTEMC_VERSION}"; \
      python3 -c "import sys, numpy, yaml, torch; print('python  ', sys.version.split()[0]); print('numpy   ', numpy.__version__); print('pyyaml  ', yaml.__version__); print('torch   ', torch.__version__)"; \
    } > /etc/ca_fp_env_versions.txt; \
    cat /etc/ca_fp_env_versions.txt

# Also wire up `conda activate`/`conda deactivate` and the "(ca_fp_2026)"
# prompt prefix for anyone poking around interactively -- purely a UX nicety
# on top of the PATH fix above, not load-bearing for it.
RUN echo "source ${CONDA_HOME}/etc/profile.d/conda.sh" >> /etc/bash.bashrc \
    && echo "conda activate ca_fp_2026" >> /etc/bash.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
