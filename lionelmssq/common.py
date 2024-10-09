import re


_NUCLEOSIDE_RE = re.compile(r"\d*[ACGU]")


def parse_nucleosides(sequence: str):
    return _NUCLEOSIDE_RE.findall(sequence)
