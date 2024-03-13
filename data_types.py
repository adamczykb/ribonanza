from enum import Enum
from dataclasses import dataclass


class ResidueType(Enum):
    ADEINE = 1
    THYMINE = 2
    URACIL = 3
    GUANINE = 4


@dataclass
class SequenceEntity:
    residue: ResidueType
    pairing_depth: int
    value: float

    def __str__(self):
        return f"{0} {1} {2}".format(self.residue, self.pairing_depth, self.value)

    def getOneHot(self):
        result = []
        result.extend([0 if i != self.residue else 1 for i in range(4)])
        result.extend([0 if i != self.pairing_depth else 1 for i in range(5)])
        return result


@dataclass
class Sequence:
    sequence: list[SequenceEntity]
    sequence_id:str

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return str([str(i) for i in self.sequence])
