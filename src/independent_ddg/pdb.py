"""Small, deterministic PDB sequence and author-number mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "HSD": "H", "HSE": "H", "HSP": "H", "CYX": "C", "CYN": "C",
}


@dataclass(frozen=True)
class Residue:
    chain: str
    resseq: int
    icode: str
    name3: str
    aa: str


@dataclass(frozen=True)
class CoordinateResidue:
    chain: str
    resseq: int
    icode: str
    name3: str
    aa: str
    ca: tuple[float, float, float]


def read_atom_residues(path: str | Path) -> dict[str, list[Residue]]:
    """Read the first model and return unique ATOM residues in file order."""

    chains: dict[str, list[Residue]] = {}
    seen: set[tuple[str, int, str]] = set()
    in_first_model = True
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_text = line[10:14].strip()
                in_first_model = not model_text or model_text == "1"
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if record != "ATOM" or not in_first_model or len(line) < 27:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            name3 = line[17:20].strip().upper()
            chains.setdefault(chain, []).append(
                Residue(chain, resseq, icode, name3, THREE_TO_ONE.get(name3, "X"))
            )
    return chains


def sequence_and_index(residues: list[Residue]) -> tuple[str, dict[tuple[int, str], int]]:
    sequence = "".join(residue.aa for residue in residues)
    index = {(residue.resseq, residue.icode): position for position, residue in enumerate(residues)}
    return sequence, index


def read_ca_residues(path: str | Path) -> list[CoordinateResidue]:
    """Read one C-alpha coordinate per residue from the first PDB model."""

    residues: list[CoordinateResidue] = []
    seen: set[tuple[str, int, str]] = set()
    in_first_model = True
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_text = line[10:14].strip()
                in_first_model = not model_text or model_text == "1"
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if record != "ATOM" or not in_first_model or len(line) < 54:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            name3 = line[17:20].strip().upper()
            residues.append(
                CoordinateResidue(
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    name3=name3,
                    aa=THREE_TO_ONE.get(name3, "X"),
                    ca=(x, y, z),
                )
            )
    return residues
