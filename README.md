# Hamiltonian-Guided Autoregressive Selected-Configuration Interaction(HAAR-SCI) Achieves Chemical Accuracy in Strongly Correlated Systems

The current project temporarily named as Quantum-Many-Body (`qmp`) which is a powerful tool designed to solve quantum-many-body problems especially for strongly correlated systems.

## About The Project

This repository hosts a [Python][python-url] package named `qmp`, dedicated to solving quantum-many-body problem.
It implements a suite of algorithms and interfaces with various model descriptors, such as the [OpenFermion][openfermion-url] format and FCIDUMP.
Additionally, `qmp` can efficiently utilize accelerators such as GPU(s) to enhance its performance.
The package's main entry point is a command line interface (CLI) application, also named `qmp`.

## Getting Started

Users can run this application either using [Docker][docker-url] or locally.
Both approaches require GPU(s) with [CUDA][cuda-url] support and a properly installed GPU driver, which is typically included with the installation of the CUDA Toolkit.

### Run with Docker

After installing [Docker][docker-url] with [CUDA support][docker-cuda-url], pull [our prebuilt Docker image][our-docker-url] using:
```
docker pull hzhangxyz/qmp
```
If users experience network issues, consider [configuring Docker mirrors][docker-mirror-url].

Then, user can run `qmp` with
```
docker run --device=nvidia.com/gpu=all --rm -it hzhangxyz/qmp --help
```

This command utilizes Docker's [CDI][docker-cuda-cdi-url] feature to enable CUDA devices in `--device=nvidia.com/gpu=all`.
Alternatively, for [legacy][docker-cuda-legacy-url] support, users can run:
```
docker run --gpus all --rm -it hzhangxyz/qmp --help
```

Please note that we currently provide Docker images for Linux/AMD64 only.

When running with Docker, users might want to [mount][docker-mount-url] a local folder to share storage between the container and the local machine such as using the `-v` option.

### Run locally

To install locally, users first needs to install the [CUDA toolkit][cuda-url].

The `qmp` requires Python >= 3.12.
After setting up a compatible Python environment such as using [Anaconda][anaconda-url], [Miniconda][miniconda-url], [venv][venv-url] or [pyenv][pyenv-url], users can install [our prebuilt package][our-pypi-url] using:
```
pip install qmp
```
If users face network issues, consider setting up a mirror with the `-i` option.

Users can then invoke the `qmp` script with:
```
qmp --help
```

Please note that if the CUDA toolkit version is too old, users must install a compatible PyTorch version before running `pip install qmp`.
For example, use `pip install torch --index-url https://download.pytorch.org/whl/cu118` for CUDA 11.8 (see [PyTorch’s guide][pytorch-install-url] for details).
This older CUDA-compatible PyTorch must be installed first, otherwise, users will need to uninstall all existing PyTorch/CUDA-related python packages before reinstalling the correct version.

## Usage

The main entry point of this package is a CLI script named `qmp`.
Use the following command to view its usage:
```
qmp --help
```

This command provides a collection of subcommands, such as `imag`.
To access detailed help for a specific subcommand, users can append `--help` to the command.
For example, use `qmp haar --help` to view the help information for the `imag` subcommand.

Typically, `qmp` requires a specific descriptor for a particular physical or chemical model to execute.
We have collected a set of such models [here][models-url].
Users can clone or download this dataset into a folder named `models` within their current working directory.
This folder `models` is the default location which `qmp` will search for the necessary model files.
Alternatively, users can specify a custom path by setting the `$QMP_MODEL_PATH` environment variable, thereby overriding the default behavior.

After cloning or downloading the dataset, users can calculate the ground state of the $N_2$ system by running the command:
```
qmp haar openfermion mlp -PN2
```
This command utilizes the `imag` subcommand with the descriptor in OpenFermion format and the [mlp network][naqs-url],
It specifies the $N_2$ model via the `-PN2` flag since the $N_2$ model is loaded from the file `N2.hdf5` in the folder `models`.

For more detailed information, please refer to the help command and the documentation.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is distributed under the GPLv3 License. See [LICENSE.md](LICENSE.md) for more information.

[python-url]: https://www.python.org/
[openfermion-url]: https://quantumai.google/openfermion
[docker-url]: https://www.docker.com/
[cuda-url]: https://docs.nvidia.com/cuda/
[docker-cuda-url]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
[our-docker-url]: https://hub.docker.com/r/hzhangxyz/qmp
[docker-mirror-url]: https://docs.docker.com/docker-hub/image-library/mirror/
[docker-cuda-cdi-url]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
[docker-cuda-legacy-url]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[anaconda-url]: https://www.anaconda.com/
[miniconda-url]: https://docs.anaconda.com/miniconda/
[venv-url]: https://docs.python.org/3/library/venv.html
[pyenv-url]: https://github.com/pyenv/pyenv
[our-pypi-url]: https://pypi.org/project/qmp/
[docker-mount-url]: https://docs.docker.com/engine/storage/volumes/
[pytorch-install-url]: https://pytorch.org/get-started/locally/
[models-url]: https://huggingface.co/datasets/USTC-KnowledgeComputingLab/qmp-models
[naqs-url]: https://github.com/tomdbar/naqs-for-quantum-chemistry
