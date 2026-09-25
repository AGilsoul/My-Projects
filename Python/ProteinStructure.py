from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf


class AminoAcid:
    def __init__(self, name: str, code: str, polar: bool,  charge: int):
        self.name = name
        self.code = code
        self.polar = polar
        self.charge = charge

    def __str__(self):
        res = f'({self.name}: {self.code}, {"polar" if self.polar else "non polar"}, charge: {self.charge})'
        return res


# First I will look into self avoiding walk with FCC structure!

AMINO_ACIDS = {
    'R': AminoAcid('Arginine', 'R', True, 1),
    'H': AminoAcid('Histidine', 'H', True, 1),
    'K': AminoAcid('Lysine', 'K', True, 1),
    'D': AminoAcid('Aspartic Acid', 'D', True, -1),
    'E': AminoAcid('Glutamic Acid', 'E', True, -1),
    'S': AminoAcid('Serine', 'S', True, 0),
    'T': AminoAcid('Threonine', 'T', True, 0),
    'N': AminoAcid('Aspargine', 'N', True, 0),
    'Q': AminoAcid('Glutamine', 'Q', True, 0),
    'C': AminoAcid('Cysteine', 'C', False, 0),
    'G': AminoAcid('Glycine', 'G', True, 0),
    'P': AminoAcid('Proline', 'P', False, 0),
    'A': AminoAcid('Alanine', 'A', False, 0),
    'V': AminoAcid('Valine', 'V', False, 0),
    'I': AminoAcid('Isoleucine', 'I', False, 0),
    'L': AminoAcid('Leucine', 'L', False, 0),
    'M': AminoAcid('Methionine', 'M', False, 0),
    'F': AminoAcid('Phenylalanine', 'F', False, 0),
    'Y': AminoAcid('Tyrosine', 'Y', False, 0),
    'W': AminoAcid('Tryptophan', 'W', False, 0)
}


