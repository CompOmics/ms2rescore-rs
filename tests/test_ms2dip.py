"""Tests for ms2dip_extract_intensity_grid."""

import math

import numpy as np
from ms2rescore_rs import (
    MS2Spectrum,
    Precursor,
    annotate_ms2_spectra,
    ms2dip_extract_intensity_grid,
)

# Known b-ion m/z values for PEPTIDE (monoisotopic, charge 1), reused from test_annotation.py:
# b1: P  = 98.06004
# b2: PE = 227.10263
# b3: PEP = 324.15540

PEPTIDE_PROFORMA = "PEPTIDE"
PEPTIDE_SEQ_LEN = 7  # 7 amino acids = 6 possible b/y ions


def _make_spectrum(identifier, mz, intensity, charge=2):
    precursor = Precursor(mz=500.0, charge=charge)
    return MS2Spectrum(identifier=identifier, mz=mz, intensity=intensity, precursor=precursor)


class TestMs2dipExtractIntensityGrid:
    def test_end_to_end_with_annotate_ms2_spectra(self):
        """Peaks matching b2 and b3 of PEPTIDE should land in the expected grid cells."""
        spectrum = _make_spectrum("test1", [227.1026, 324.1554], [100.0, 200.0])

        annotated = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        max_len = 10
        grids = ms2dip_extract_intensity_grid(
            annotated_spectra=annotated,
            seq_lens=[PEPTIDE_SEQ_LEN],
            precursor_charges=[2],
            row_ion_types=["b", "y", "b", "y"],
            row_charges=[1, 1, 2, 2],
            reverse_ions=["y"],
            max_peptide_length=max_len,
        )

        assert len(grids) == 1
        grid = np.asarray(grids[0])
        assert grid.shape == (4, max_len)

        # n_cols = min(7, 10) - 1 = 6 possible columns (0..5), rest NaN, for every row
        # (precursor charge 2 unmasks both charge-1 and charge-2 rows).
        for row in range(4):
            for col in range(6):
                assert not math.isnan(grid[row, col])
            for col in range(6, max_len):
                assert math.isnan(grid[row, col])

        # Row 0 = b, charge 1: b2 -> col 1 -> 100.0; b3 -> col 2 -> 200.0
        assert grid[0, 1] == 100.0
        assert grid[0, 2] == 200.0
        # Untouched in-range b1 cell stays at the 0.0 baseline.
        assert grid[0, 0] == 0.0

        # Row 2 = b, charge 2: nothing matched charge 2, all in-range cells at baseline.
        assert np.all(grid[2, :6] == 0.0)

    def test_mismatched_lengths_raise(self):
        spectrum = _make_spectrum("test1", [227.1026], [100.0])
        annotated = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )
        try:
            ms2dip_extract_intensity_grid(
                annotated_spectra=annotated,
                seq_lens=[PEPTIDE_SEQ_LEN, PEPTIDE_SEQ_LEN],  # length mismatch on purpose
                precursor_charges=[2],
                row_ion_types=["b"],
                row_charges=[1],
                reverse_ions=[],
                max_peptide_length=10,
            )
            raise AssertionError("expected a ValueError")
        except ValueError:
            pass
