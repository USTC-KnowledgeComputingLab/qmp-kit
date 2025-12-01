import typing
import re
import gzip
import pathlib
import logging
import dataclasses
import torch
from .common import CommonConfig
from .hamiltonian import Hamiltonian
from .subcommand_dict import subcommand_dict


class NaturalOrbitCalculator:

    def __init__(self, orbit: int):
        self.orbit = orbit
        self.hopping = [[Hamiltonian({
            ((2 * i, 1), (2 * j, 0)): 1,
            ((2 * i + 1, 1), (2 * j + 1, 0)): 1,
        }, kind="fermi") for j in range(orbit)] for i in range(orbit)]

    def matrix(self, config: torch.Tensor, psi: torch.Tensor):
        return torch.tensor([[psi.conj() @ self.hopping[i][j].apply_within(config, psi, config) for j in range(self.orbit)] for i in range(self.orbit)])

    def unitary(self, config: torch.Tensor, psi: torch.Tensor):
        mat = self.matrix(config, psi)
        eigvals, eigvecs = torch.linalg.eigh(mat)
        return eigvecs


@dataclasses.dataclass
class NaturalOrbit:

    common: CommonConfig

    config_psi: str
    fcidump_src: str
    fcidump_dst: str

    def main(self, *, model_param: typing.Any = None, network_param: typing.Any = None) -> None:

        model, network, _ = self.common.main(model_param=model_param, network_param=network_param)

        data = torch.load(self.config_psi)
        config = data["config"]
        psi = data["psi"]
        orbit_number = model.n_qubit // 2
        logging.info("data loaded")
        calculator = NaturalOrbitCalculator(orbit_number)
        U = calculator.unitary(config, psi)
        logging.info("unitary matrix calculated")

        with (
                gzip.open(self.fcidump_src, "rt", encoding="utf-8") if self.fcidump_src.endswith(".gz") else open(self.fcidump_src, "rt", encoding="utf-8") as src,
                gzip.open(self.fcidump_dst, "wt", encoding="utf-8") if self.fcidump_dst.endswith(".gz") else open(self.fcidump_dst, "wt", encoding="utf-8") as dst,
        ):
            n_orbit: int | None = None
            n_electron: int | None = None
            n_spin: int | None = None
            for line in src:
                print(line, file=dst)
                data: str = line.lower()
                if (match := re.search(r"norb\s*=\s*(\d+)", data)) is not None:
                    n_orbit = int(match.group(1))
                if (match := re.search(r"nelec\s*=\s*(\d+)", data)) is not None:
                    n_electron = int(match.group(1))
                if (match := re.search(r"ms2\s*=\s*(\d+)", data)) is not None:
                    n_spin = int(match.group(1))
                if "&end" in data:
                    break
            energy_0: float = 0.0
            energy_1: torch.Tensor = torch.zeros([n_orbit, n_orbit], dtype=torch.float64)
            energy_2: torch.Tensor = torch.zeros([n_orbit, n_orbit, n_orbit, n_orbit], dtype=torch.float64)
            for line in src:
                pieces: list[str] = line.split()
                coefficient: float = float(pieces[0])
                sites: tuple[int, ...] = tuple(int(i) - 1 for i in pieces[1:])
                match sites:
                    case (-1, -1, -1, -1):
                        energy_0 = coefficient
                    case (_, -1, -1, -1):
                        # Psi4 writes additional non-standard one-electron integrals in this format, which we omit.
                        pass
                    case (i, j, -1, -1):
                        energy_1[i, j] = coefficient
                        energy_1[j, i] = coefficient
                    case (_, _, _, -1):
                        # In the standard FCIDUMP format, there is no such term.
                        raise ValueError(f"Invalid FCIDUMP format: {sites}")
                    case (i, j, k, l):
                        energy_2[i, j, k, l] = coefficient
                        energy_2[i, j, l, k] = coefficient
                        energy_2[j, i, k, l] = coefficient
                        energy_2[j, i, l, k] = coefficient
                        energy_2[l, k, j, i] = coefficient
                        energy_2[k, l, j, i] = coefficient
                        energy_2[l, k, i, j] = coefficient
                        energy_2[k, l, i, j] = coefficient
                    case _:
                        raise ValueError(f"Invalid FCIDUMP format: {sites}")
            energy_1 = energy_1.to(dtype=torch.complex128)
            energy_2 = energy_2.to(dtype=torch.complex128)
            energy_1 = U.conj().T @ energy_1 @ U
            # The matrix is too large, so we use einsum step by step.
            # energy_2 = torch.einsum("pi,qj,rk,sl,ijkl->pqrs", U.conj(), U.conj(), U, U, energy_2)
            temp = torch.einsum("rk,ijkl->ijrl", U, energy_2)
            temp = torch.einsum("sl,ijrl->ijrs", U, temp)
            temp = torch.einsum("qj,ijrs->iqrs", U.conj(), temp)
            energy_2 = torch.einsum("pi,iqrs->pqrs", U.conj(), temp)

            print(f" {energy_0:22.15e}   0  0  0  0", file=dst)
            for i in range(n_orbit):
                for j in range(n_orbit):
                    if abs(energy_1[i, j]) > 1e-12:
                        print(f" {energy_1[i, j]:22.15e}   {i+1}  {j+1}  0  0", file=dst)
            for i in range(n_orbit):
                for j in range(n_orbit):
                    for k in range(n_orbit):
                        for l in range(n_orbit):
                            if abs(energy_2[i, j, k, l]) > 1e-12:
                                print(f" {energy_2[i, j, k, l]:22.15e}   {i+1}  {j+1}  {k+1}  {l+1}", file=dst)


subcommand_dict["naturalorbit"] = NaturalOrbit
