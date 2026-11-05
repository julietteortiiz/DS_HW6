"""HW06 Shannon coding; adapted from Allison Gong's original course code."""
from __future__ import annotations
import math


class Shannon:
    """A cumulative-probability Shannon code; computations never print.

    Preserve input order for equal probabilities. Copy the probability mapping
    so constructing a model does not mutate it. Store strings, including zeros.
    """
    mapping: dict[str, float]
    binary_codes: dict[str, str]

    def __init__(self, char_probs: dict[str, float]) -> None:
        """Build a codebook from positive finite probabilities summing to one.

        Keys must be single characters. Reject an empty/invalid distribution
        with ValueError (sum tolerance: absolute 1e-9). A singleton uses '0'.
        """
        raise NotImplementedError("TODO: implement __init__")

    def get_binary_codes(self) -> dict[str, str]:
        """Return a copy of the character-to-codeword dictionary."""
        return self.binary_codes.copy()

    def calc_num_digits(self, prob: float) -> int:
        """Return ceil(-log2(prob)), with minimum 1 for the singleton case.

        Raise ValueError unless prob is finite and 0 < prob <= 1.
        """
        raise NotImplementedError("TODO: implement calc_num_digits")

    def convert_binary(self, cum_prob: float, num_digits: int) -> str:
        """Return exactly num_digits fractional binary digits of cum_prob.

        Truncate, never round; preserve leading/trailing zeros. Require finite
        0 <= cum_prob < 1 and num_digits >= 0, otherwise raise ValueError.
        The explicit length bounds the loop even for nonterminating fractions.
        """
        raise NotImplementedError("TODO: implement convert_binary")

    def binary_correct_digits(self, binary: str, num_digits: int) -> str:
        """Truncate to num_digits or pad on the right with zeros.

        Reject nonbinary input or negative num_digits with ValueError.
        """
        raise NotImplementedError("TODO: implement binary_correct_digits")

    def calc_binary(self) -> dict[str, str]:
        """Return a newly constructed codebook using self.mapping.

        Sort by decreasing probability, retaining dictionary order in ties.
        Use cumulative probability BEFORE the current character and the length
        returned by calc_num_digits. Do not modify the probability dictionary.
        """
        raise NotImplementedError("TODO: implement calc_binary")

    def encode(self, text: str) -> str:
        """Concatenate codewords; empty text returns ''. Unknown chars: KeyError."""
        raise NotImplementedError("TODO: implement encode")

    def decode(self, bits: str) -> str:
        """Decode complete codewords; empty input returns ''.

        Raise ValueError for nonbinary digits, impossible prefixes, or an
        unfinished final codeword. Do not strip whitespace or ignore bad bits.
        """
        raise NotImplementedError("TODO: implement decode")

    def expected_length(self) -> float:
        """Return source-weighted bits per character, not encoded-file bytes."""
        raise NotImplementedError("TODO: implement expected_length")
