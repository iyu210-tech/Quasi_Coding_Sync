from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
from fibonacci_qc import config as cfg

def fibonacci_number(n:int) -> int:
    if n<1:
        raise ValueError("Fib. index must be >= 1.")
    if n == 1 or n == 2:
        return 1
    a, b = 1, 1
    for k in range(3, n + 1):
        a, b = b, a + b
    return b

def inflate_word(word:str) -> str:
    parts = []
    for letter in word:
        try:
            parts.append(cfg.SUBSTITUTION[letter])
        except KeyError as exc:
            raise ValueError(f"Unknown letter '{letter}' in word.") from exc
    return "".join(parts)

def fibonacci_word(n:int) -> str:
    "Generate W_n and verify |W_n| = F_n."
    if n < 1:
        raise ValueError("Fib. index must be >= 1.")
    if n == 1:
        word = cfg.WORD_W1
    elif n == 2:
        word = cfg.WORD_W2
    else:
        prev2 = cfg.WORD_W1
        prev1 = cfg.WORD_W2
        for k in range(3, n + 1):
            prev2, prev1 = prev1, prev1 + prev2
        word = prev1
        verify_word(n, word)
    return word

def verify_word(n:int, word:str) -> None:
    "Raise ValueError if word is not the locked word W_n"
    expected_len = fibonacci_number(n)
    if len(word) != expected_len:
        raise ValueError(f"Word length {len(word)} does not match F_{n} = {expected_len}.")
    illegal = set(word)-set("AB")
    if illegal:
        raise ValueError(f"Word contains illegal letters: {sorted(illegal)}.")
    if "BB" in word:
        raise ValueError("Word contains illegal substring 'BB'.")
    if n == 1 and word != cfg.WORD_W1:
        raise ValueError(f"W_1 must be {cfg.WORD_W1 !r}, got {word !r}.")
    if n == 2 and word != cfg.WORD_W2:
        raise ValueError(f"W_2 must be {cfg.WORD_W2 !r}, got {word !r}.")
    if n >= 3:
        left = _fibonacci_word_unchecked(n - 1)
        right = _fibonacci_word_unchecked(n - 2)
        if word != left + right:
            raise ValueError(f"W_{n} does not equal W_{n-1} + W_{n-2}")
        if word != inflate_word(left):
            raise ValueError(f"W_{n} is not consistent with the substitution applied to W_{n-1}.")


def _fibonacci_word_unchecked(n:int) -> str:
    "Create W_n without complete verification. Only will be used internally."
    if n == 1:
        return cfg.WORD_W1
    if n == 2:
        return cfg.WORD_W2
    prev2 = cfg.WORD_W1
    prev1 = cfg.WORD_W2
    for k in range(3, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1

def centered_site_window(letters: np.ndarray, p: int, radius: int, left_token: str = cfg.LEAD_LEFT_TOKEN, right_token: str = cfg.LEAD_RIGHT_TOKEN) -> str:
    "Return (2*radius+1) letter long motif centered on site p."
    if radius < 0: 
        raise ValueError("Radius cannot be negative.")
    n_sites = letters.size
    chars = []
    for q in range(p - radius, p + radius + 1):
        if q < 0:
            chars.append(left_token)
        elif q >= n_sites:
            chars.append(right_token)
        else:
            chars.append(letters[q])
    return "".join(chars)

def bond_window(letters: np.ndarray, p: int, radius: int, left_token: str = cfg.LEAD_LEFT_TOKEN, right_token: str = cfg.LEAD_RIGHT_TOKEN) -> str:
    "Return bond motif whose left site is p."
    #radius == 0 is the pair of two letters at (p,p+1)
    if radius < 0:
        raise ValueError("Radius cannot be negative.")
    n_sites = letters.size
    if p < 0 or p >= n_sites - 1:
        raise IndexError(f"bond index p={p} is out of range for N={n_sites}.")
    chars = []
    for q in range(p - radius, p + radius + 2):
        if q < 0:
            chars.append(left_token)
        elif q >= n_sites:
            chars.append(right_token)
        else:
            chars.append(letters[q])
    return "".join(chars)


def _motif_array(letters: np.ndarray, n_positions:int, radius: int, kind: str) -> np.ndarray:
    width = (2*radius+1) if kind == "site" else (2*radius + 2)
    out = np.empty(n_positions, dtype=f"<U{width}")
    if kind == "site":
        for p in range(n_positions):
            out[p] = centered_site_window(letters, p, radius)
    else:
        for p in range(n_positions):
            out[p] = bond_window(letters, p, radius)
    return out

@dataclasses(frozen=True)
class FibonacciGeometry:
    """Fib stack at one generation, with geometry labels
    Array lengths:
        sites: N = F_n = |W_n|
        bonds: N-1, indexed by left site of pair
    """

    generation: int
    word: str
    letters: np.ndarray
    widths_nm: np.ndarray
    potentials_ev: np.ndarray
    site_index: np.ndarray
    site_parity: np.ndarray
    site_motifs: Dict[str, np.ndarray]
    bond_index: np.ndarray
    bond_parity: np.ndarray
    bond_pairs: np.ndarray
    phason_valid: np.ndarray
    bond_motifs: Dict[str, np.ndarray]

    @property
    def n_sites(self) -> int:
        return int(self.letters.size)

    @property
    def n_bonds(self) -> int:
        return int(self.bond_index.size)

    def site_labels(self, family: str = "site_letter") -> np.ndarray:
        try:
            return self.site_motifs[family]
        except KeyError as exc:
            raise KeyError(
                f"Unkonwn site motif family {family!r};"
                f"known: {tuple(self.site_motifs)}"
            ) from exc

    def bond_labels(self, family: str = "bond_pair") -> np.ndarray:
        try:
            return self.bond_motifs[family]
        except KeyError as exc:
            raise KeyError(
                f"Unkonwn bond motif family {family!r};"
                f"known: {tuple(self.bond_motifs)}"
            ) from exc

    def motif_labels(self, family: str) -> np.ndarray:
        if family in self.site_motifs:
            return self.site_labels(family)
        elif family in self.bond_motifs:
            return self.bond_labels(family)
        raise KeyError(
            f"Unkonwn motif family {family!r}; known: {cfg.MOTIF_FAMILIES}"
        )

    def stratum_labels(self, family: str) -> np.ndarray:
        return self.motif_labels(family)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "generation": self.generation,
            "n_sites": self.n_sites,
            "n_bonds": self.n_bonds,
            "fibonacci_number": fibonacci_number(self.generation),
            "n_phason_valid": int(np.count_nonzero(self.phason_valid)),
            "n_phason_invalid": int(self.n_bonds - np.count_nonzero(self.phason_valid)),
            "site_motif_families": tuple(self.site_motifs),
            "bond_motif_families": tuple(self.bond_motifs),
        }


def map_letters_to_layers(letters: np.ndarray, width_map: Mapping[str,float] = cfg.LAYER_WIDth_NM, potential_map: Mapping[str,float] = cfg.LAYER_POTENTIAL_EV,
) -> Tuple[np.ndarray, np.ndarray]:
    "Map AB letters onto widths and potentials"
    widths = np.empty(letters.shape, dtype=np.float64)
    pots = np.empty(letters.shape, dtype=np.float64)
    for letter, width in width_map.items():
        mask = letters == letter
        widths[mask] = width
        pots[mask] = potential_map[letter]
    unknown = ~np.isin(letters, list(width_map))
    if np.any(unknown):
        bad = sorted(set(letters[unknown].tolist()))
        raise ValueError(f"cannot map letters {bad} to layers")
    return widths, pots

def build_geometry(n:int) -> FibonacciGeometry:
    "Build the stack for generation n"
    word = fibonacci_word(n)
    letters = np.array(list(word),dtype="<U1")
    n_sites = letters.size
    widths, pots = map_letters_to_layers(letters)

    site_index = np.arrange(n_sites, dtype=np.int64)
    site_parity = (site_index % 2).astype(np.int8)
    site_motifs = {
        family: _motif_array(letters, n_sites, radius, kind = "site")
        for family, radius in cfg.SITE_MOTIF_RADIUS.items()
    }

    n_bonds = max(n_sites - 1, 0)
    bond_index = np.arrage(n_bonds, dtype = np.int64)
    bond_parity = (bond_index % 2).astype(np.int8)
    if n_bonds:
        bond_pairs = np.array(
            [str(letters[p]) + str(letters[p+1]) for p in range(n_bonds)],
            dtype="<U2"
        )
        phason_valid = np.isin(bond_pairs, np.array(["AB", "BA"]))
    else:
        bond_pairs = np.empty(0, dtype = "<U2")
        phason_valid = np.empty(0,dtype=bool)

    bond_motifs = {
        family: _motif_array(letters, n_bonds, radius, kind="bond")
        for family, radius in cfg.BOND_MOTIF_RADIUS.items()
    }
    if n_bonds:
        if not np.array_equal(bond_motifs["bond_pair"], bond_pairs):
            raise RuntimeError("bond_pair motifs disagree with bond_pairs")

    return FibonacciGeometry(
        generation=n
        word=word
        letters=letters
        widths_nm=widths
        potentials_ev=pots
        site_index=site_index,
        site_parity=site_parity,
        site_motifs=site_motifs,
        bond_index=bond_index,
        bond_parity=bond_parity
        bond_pairs=bond_pairs,
        phason_valid=phason_valid,
        bond_motifs=bond_motifs
    )

def iter_study_geometries(
    generations: Optional[Iterable[int]] = None,
) -> Iterator[FibonacciGeometry]:
    "Labeled stacks for the generations"
    if generations is None:
        generations = cfg.GENERATIONS
    for n in generations:
        yield build_geometry(n)