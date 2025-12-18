# Quantum Many-Body Problem Kit (qmp-kit)

The quantum many-body problem kit (`qmp-kit`) is a powerful tool designed to solve quantum-many-body problems especially for strongly correlated systems. This project includes our work on [Hamiltonian-Guided Autoregressive Selected-Configuration Interaction Achieves Chemical Accuracy in Strongly Correlated Systems](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415).

## About The Project

This repository hosts a [Python][python-url] package named `qmp`, dedicated to solving quantum-many-body problem.
It implements a suite of algorithms and interfaces with various model descriptors, such as the [OpenFermion][openfermion-url] format and FCIDUMP.
Additionally, `qmp` can efficiently utilize accelerators such as GPU(s) to enhance its performance.
The package's main entry point is a command line interface (CLI) application, also named `qmp`.

## Getting Started

To run this application locally, you need GPU(s) with [CUDA][cuda-url] support and a properly installed GPU driver (typically included with the CUDA Toolkit installation).

### Local Installation

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
[cuda-url]: https://docs.nvidia.com/cuda/
[anaconda-url]: https://www.anaconda.com/
[miniconda-url]: https://docs.anaconda.com/miniconda/
[venv-url]: https://docs.python.org/3/library/venv.html
[pyenv-url]: https://github.com/pyenv/pyenv
[our-pypi-url]: https://pypi.org/project/qmp/
[pytorch-install-url]: https://pytorch.org/get-started/locally/
[models-url]: https://huggingface.co/datasets/USTC-KnowledgeComputingLab/qmp-models
[naqs-url]: https://github.com/tomdbar/naqs-for-quantum-chemistry
