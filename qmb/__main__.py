"""
This is the main entry point for the command line application.

For the details of the command line application, run `qmb --help` or `python -m qmb --help`.
"""

import tyro
from . import openfermion as _  # type: ignore[no-redef]
from . import fcidump as _  # type: ignore[no-redef]
from . import hubbard as _  # type: ignore[no-redef]
from . import free_fermion as _  # type: ignore[no-redef]
from . import ising as _  # type: ignore[no-redef]
from . import vmc as _  # type: ignore[no-redef]
from . import markov as _  # type: ignore[no-redef]
from . import haar as _  # type: ignore[no-redef]
from . import rldiag as _  # type: ignore[no-redef]
from . import precompile as _  # type: ignore[no-redef]
from . import list_loss as _  # type: ignore[no-redef]
from . import chop_imag as _  # type: ignore[no-redef]
from . import pert as _  # type: ignore[no-redef]
from . import run as _  # type: ignore[no-redef]
from .subcommand_dict import subcommand_dict


def main() -> None:
    """
    The main function for the command line application.
    """
    tyro.extras.subcommand_cli_from_dict(subcommand_dict).main()


if __name__ == "__main__":
    main()
