from __future__ import annotations

import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import(
    bond_window,
    build_geometry,
    centered_site_window,
    fibonacci_number,
    fibonacci_word,
    inflate_word,
    iter_study_geometries,
    map_letters_to_layers,
    verify_word,
)

KNOWN_WORDS = {
    1:"B",
    2:"A",
    3:"AB",
    4:"ABA",
    5:"ABAAB",
    6:"ABAABABA",
    7:"ABAABABAABAAB",
}

KNOWN_F = {
    1:1,
    2:1,
    3:2,
    4:3,
    5:5,
    6:8,
    7:13,
    10:55,
    15:610,
    20:6765
}

@pytest.mark.parametrize("n, expected", sorted(KNOWN_F.items()))
def test_fibonacci_numbers(n,expected):
    assert fibonacci_number(n) == expected

def test_fibonacci_recurrence_and_length():
    for n in range(1,21):
        word = fibonacci_word(n)
        assert len(word) == fibonacci_number(n)
        if n>= 3:
            assert word == fibonacci_word(n-1) + fibonacci_word(n-2)
            assert word == inflate_word(fibonacci_word(n-1))

@pytest.mark.parametrize("n, expected", sorted(KNOWN_WORDS.items()))
def test_known_words(n, expected):
    assert fibonacci_word(n) == expected
    verify_word(n, expected)

def test_verify_word_rejects_length_mismatch():
    with pytest.raises(ValueError, match = "does not equal"):
        verify_word(5,"AB")

def test_physical_layer_mapping():
    letters = np.array(list("ABAB"),dtype="<U1")
    widths, pots = map_letters_to_layers(letters)
    np.testing.assert_allclose(widths, [1.0,2.0,1.0,2.0])
    np.testing.assert_allclose(pots, [0.01, 5.0,0.01, 5.0])

def test_generation_five_labels_and_uncompacted_phason_mask():
    geom = build_geometry(5)
    assert geom.word == "ABAAB"
    assert geom.n_sites == 5
    assert geom.n_bonds == 4
    np.testing.assert_array_equal(geom.letters, list("ABAAB"))
    np.testing.assert_allclose(geom.widths_nm, [1.0,2.0,1.0,1.0,2.0])
    np.testing.assert_allclose(geom.potentials_ev, [0.01,5.0,0.01,0.01,5.0])
    np.testing.assert_array_equal(geom.site_index, [0,1,2,3,4])
    np.testing.assert_array_equal(geom.site_parity, [0,1,0,1,0])
    np.testing.assert_array_equal(geom.bond_pairs, ["AB","BA","AA","AB"])
    np.testing.assert_array_equal(geom.phason_valid, [True, True, False, True])
    assert geom.phason_valid.shape == (geom.n_bonds)
    #AA at index 2 stays false, mark not compacted
    assert geom.bond_index[2]==2
    assert geom.phason_valid[2] == False
    np.testing.assert_array_equal(geom.site_labels("site_letter"),list("ABAAB"))
    np.testing.assert_array_equal(geom.bond_labels("bond_pair"), geom.bond_pairs)

def test_boundary_motifs_use_lead_tokens():
    geom = build_geometry(5)
    letters = geom.letters
    assert centered_site_window(letters,0,1) == "LAB"
    assert centered_site_window(letters,4,1) == "ABR"
    assert centered_site_window(letters,0,2) == "LLABA"
    assert centered_site_window(letters,2,2) == "ABAAB"
    assert bond_window(letters,0,0) == "AB"
    assert bond_window(letters,0,1) == "LABA"
    assert bond_window(letters,3,1) == "AABR"
    np.testing.assert_array_equal(
        geom.site_motifs["site_3"], ["LAB","ABA","BAA","AAB","ABR"])

def test_study_generations_match_locked_range_and_fn():
    built = list(iter_study_geometries())
    assert [g.generation for g in built] == list(cfg.GENERATIONS)
    for geom in built:
        assert geom.n_sites == fibonacci_number(geom.generation)
        assert geom.n_bonds == geom.n_sites - 1
        assert geom.phason_valid.shape == (geom.n_bonds,)
        assert geom.site_parity.shape == (geom.n_sites,)
        assert set(np.unique(geom.letters)).issubset({"A","B"})
        assert "BB" not in geom.word
        a_mask = geom.letters == "A"
        b_mask = geom.letters == "B"
        assert np.all(geom.widths_nm[a_mask] == cfg.D_A_NM)
        assert np.all(geom.widths_nm[b_mask] == cfg.D_B_NM)
        assert np.all(geom.potential_ev[a_mask] == cfg.V_A_EV)
        assert np.all(geom.potential_ev[b_mask] == cfg.V_B_EV)
        for family in cfg.SITE_MOTIF_FAMILIES:
            assert geom.site_labels(family).shape == (geom.n_sites,)
        for family in cfg.BOND_MOTIF_FAMILIES:
            assert geom.bond_labels(family).shape == (geom.n_bonds,)
        for family in cfg.GEOMETRY_STRATA:
            labels = geom.stratum_labels(family)
            expected_len = geom.n_sites if family.startswith("site_") else geom.n_bonds
            assert labels.shape == (expected_len,)

def test_no_bb_and_phason_only_on_ab_ba():
    for n in range(2,21):
        geom = build_geometry(n)
        assert "BB" not in geom.word
        np.testing.assert_array_equal(
            geom.phason_valid,
            np.isin(geom.bond_pairs, np.array(["AB","BA"])),
        )
        assert np.all(geom.bond_pairs[geom.phason_valid] != "AA")
        n_false = int(np.count_nonzero(~geom.phason_valid))
        n_aa = int(np.count_nonzero(geom.bond_pairs == "AA"))
        assert n_false == n_aa

def test_site_5_radius_length_and_labels():
    assert cfg.SITE_MOTIF_RADIUS["site_5"] == 2
    assert cfg.SITE_MOTIF_RADIUS["site_3"] == 1
    geom = build_geometry(5)
    site_3 = geom.site_labels("site_3")
    site_5 = geom.site_labels("site_5")
    assert site_3.shape == (geom.n_sites,)
    assert site_5.shape == (geom.n_sites,)
    assert all(len(s) == 2 * 1 + 1 for s in site_3)
    assert all(len(s) == 2 * 2 + 1 for s in site_5)
    assert not np.array_equal(site_3, site_5)
    letters = geom.letters
    expected_site_5 = np.array([centered_site_window(letters, p, 2) for p in range(geom.n_sites)], dtype=site_5.dtype)
    np.testing.assert_array_equal(site_5, expected_site_5)
    np.testing.assert_array_equal(site_5, ["LLABA", "LABAA", "ABAAB", "BAABR", "AABRR"])