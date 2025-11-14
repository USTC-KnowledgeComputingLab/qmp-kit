# Use the specified CUDA base image with Rocky Linux 9
FROM nvidia/cuda:13.0.2-cudnn-devel-rockylinux9

# Install dependencies
WORKDIR /app
RUN dnf update --assumeyes && \
    dnf install --assumeyes git python3.12-devel libcusparselt-devel && \
    dnf clean all
ARG PYPI_MIRROR=https://pypi.org/simple
RUN python3.12 -m ensurepip && \
    pip3.12 install --index-url=${PYPI_MIRROR} --upgrade pip

# Install the package
COPY . /app
RUN pip3.12 install --index-url=${PYPI_MIRROR} .
ENV XDG_DATA_HOME=/tmp/data XDG_CONFIG_HOME=/tmp/config XDG_CACHE_HOME=/tmp/cache XDG_STATE_HOME=/tmp/state

# Set the entrypoint
ENTRYPOINT ["qmb"]
CMD []
