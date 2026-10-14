"""Mutation notation parsing without dependencies on upstream model repositories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Mutation:
    wt: str
    chain: str
    resseq: int
    icode: str
    mutant: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PREFIXED = re.compile(
    r"^(?P<chain>[A-Za-z0-9]+):(?P<wt>[A-Z])(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)(?P<mutant>[A-Z])$"
)
_COMPACT = re.compile(
    r"^(?P<wt>[A-Z])(?P<chain>[A-Za-z0-9])(?P<resseq>-?\d+)(?P<icode>[A-Za-z]?)(?P<mutant>[A-Z])$"
)


def parse_single_mutation(text: str) -> Mutation:
    """Parse `A:D435A` or SKEMPI compact `DA435A` notation."""

    token = text.strip().replace(" ", "")
    match = _PREFIXED.match(token) or _COMPACT.match(token)
    if match is None:
        raise ValueError(f"unsupported mutation notation: {text!r}")
    fields = match.groupdict()
    return Mutation(
        wt=fields["wt"],
        chain=fields["chain"],
        resseq=int(fields["resseq"]),
        icode=fields["icode"] or "",
        mutant=fields["mutant"],
    )


def parse_mutations(text: str) -> list[Mutation]:
    tokens: Iterable[str] = (part for part in text.split(",") if part.strip())
    parsed = [parse_single_mutation(token) for token in tokens]
    if not parsed:
        raise ValueError("empty mutation list")
    return parsed

