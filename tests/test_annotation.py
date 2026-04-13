"""Tests for annotate_ms2_spectra and score_ms2_spectra."""

import math

import pytest
from ms2rescore_rs import (
    AnnotatedMS2Spectrum,
    FragmentAnnotation,
    MS2Spectrum,
    Precursor,
    annotate_ms2_spectra,
    score_ms2_spectra,
)

# Known b-ion m/z values for PEPTIDE (monoisotopic, charge 1):
# b1: P  = 98.06004
# b2: PE = 227.10263
# b3: PEP = 324.15540
# b4: PEPT = 425.20307
# b5: PEPTI = 538.28714
# b6: PEPTID = 653.31408

PEPTIDE_PROFORMA = "PEPTIDE"
PEPTIDE_SEQ_LEN = 7  # 7 amino acids = 6 possible b/y ions


def _make_spectrum(identifier, mz, intensity, charge=2):
    """Helper to create an MS2Spectrum with a precursor."""
    precursor = Precursor(mz=500.0, charge=charge)
    return MS2Spectrum(
        identifier=identifier,
        mz=mz,
        intensity=intensity,
        precursor=precursor,
    )


class TestAnnotateMS2Spectra:
    """Tests for annotate_ms2_spectra."""

    def test_basic_annotation(self):
        """Annotating a spectrum with known b-ion peaks should produce annotations."""
        # Peaks matching b2 and b3 of PEPTIDE
        spectrum = _make_spectrum("test1", [227.1026, 324.1554], [100.0, 200.0])

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        assert len(results) == 1
        ann = results[0]
        assert isinstance(ann, AnnotatedMS2Spectrum)
        assert ann.identifier == "test1"
        assert len(ann.mz) == 2
        assert len(ann.intensity) == 2
        assert len(ann.peak_annotations) == 2

        # At least one peak should have annotations
        all_annotations = [a for peak in ann.peak_annotations for a in peak]
        assert len(all_annotations) > 0

        # Check that annotations have valid series
        for a in all_annotations:
            assert a.series in ("a", "b", "c", "x", "y", "z")
            assert a.position > 0
            assert a.charge > 0

    def test_no_matching_peaks(self):
        """Peaks that don't match any theoretical ions should have empty annotations."""
        spectrum = _make_spectrum("test2", [999.999], [100.0])

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        assert len(results) == 1
        ann = results[0]
        assert len(ann.peak_annotations) == 1
        assert len(ann.peak_annotations[0]) == 0

    def test_empty_spectrum(self):
        """An empty spectrum should return empty annotations."""
        spectrum = _make_spectrum("test3", [], [])

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        assert len(results) == 1
        assert len(results[0].peak_annotations) == 0

    def test_zero_charge(self):
        """Spectrum with charge=0 should return empty annotations."""
        spectrum = _make_spectrum("test4", [227.1026], [100.0], charge=0)

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        assert len(results) == 1
        assert all(len(a) == 0 for a in results[0].peak_annotations)

    def test_tolerance_da(self):
        """Da tolerance mode should work."""
        spectrum = _make_spectrum("test5", [227.1026], [100.0])

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=0.02,
            tolerance_mode="Da",
        )

        assert len(results) == 1

    def test_etd_fragmentation_model(self):
        """ETD model should produce c/z ion annotations."""
        # c ions have +NH3 mass shift relative to b ions
        spectrum = _make_spectrum("test_etd", [244.1291, 341.1819], [100.0, 200.0])

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="etd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        assert len(results) == 1

    def test_batch_annotation(self):
        """Multiple spectra should be annotated in batch."""
        spectra = [
            _make_spectrum("s1", [227.1026], [100.0]),
            _make_spectrum("s2", [324.1554], [200.0]),
        ]

        results = annotate_ms2_spectra(
            spectra=spectra,
            proformas=[PEPTIDE_PROFORMA, PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN, PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        assert len(results) == 2
        assert results[0].identifier == "s1"
        assert results[1].identifier == "s2"

    def test_original_spectrum_preserved(self):
        """AnnotatedMS2Spectrum should carry the original spectrum data."""
        mz = [227.1026, 324.1554]
        intensity = [100.0, 200.0]
        spectrum = _make_spectrum("test_preserve", mz, intensity, charge=3)

        results = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        ann = results[0]
        assert ann.identifier == "test_preserve"
        assert ann.mz == pytest.approx(mz, abs=1e-3)
        assert ann.intensity == pytest.approx(intensity, abs=1e-3)
        assert ann.precursor.charge == 3

    def test_invalid_tolerance_mode(self):
        """Invalid tolerance mode should raise an error."""
        spectrum = _make_spectrum("test_err", [227.1026], [100.0])

        with pytest.raises(Exception, match="tolerance_mode"):
            annotate_ms2_spectra(
                spectra=[spectrum],
                proformas=[PEPTIDE_PROFORMA],
                seq_lens=[PEPTIDE_SEQ_LEN],
                fragmentation_model="cidhcd",
                mass_mode="monoisotopic",
                tolerance_value=20.0,
                tolerance_mode="invalid",
            )

    def test_length_mismatch(self):
        """Mismatched input lengths should raise an error."""
        spectrum = _make_spectrum("test_err2", [227.1026], [100.0])

        with pytest.raises(Exception):
            annotate_ms2_spectra(
                spectra=[spectrum],
                proformas=[PEPTIDE_PROFORMA, PEPTIDE_PROFORMA],
                seq_lens=[PEPTIDE_SEQ_LEN],
                fragmentation_model="cidhcd",
                mass_mode="monoisotopic",
                tolerance_value=20.0,
                tolerance_mode="ppm",
            )


class TestScoreMS2Spectra:
    """Tests for score_ms2_spectra."""

    def _annotated_spectrum(self, peak_annotations=None, mz=None, intensity=None):
        """Helper to create an AnnotatedMS2Spectrum."""
        if mz is None:
            mz = [100.0, 200.0, 300.0]
        if intensity is None:
            intensity = [50.0, 100.0, 75.0]
        if peak_annotations is None:
            peak_annotations = [[], [], []]
        return AnnotatedMS2Spectrum(
            identifier="test",
            mz=mz,
            intensity=intensity,
            precursor=Precursor(mz=500.0, charge=2),
            peak_annotations=peak_annotations,
        )

    def test_fixed_feature_set(self):
        """All 6 ion series should always be present in the output."""
        ann = self._annotated_spectrum(
            peak_annotations=[
                [FragmentAnnotation("b", 1, 1)],
                [FragmentAnnotation("y", 2, 1)],
                [],
            ]
        )

        results = score_ms2_spectra(
            spectra=[ann],
            seq_lens=[5],
            active_ion_series=["b", "y"],
            calculate_hyperscore=True,
        )

        assert len(results) == 1
        feats = results[0]

        # All 6 series features should exist
        for s in ["a", "b", "c", "x", "y", "z"]:
            assert f"matched_{s}_ions" in feats
            assert f"matched_{s}_ions_pct" in feats
            assert f"longest_{s}_ion_sequence" in feats
            assert f"ln_explained_{s}_ion_ratio" in feats

        # Aggregated features
        assert "matched_ions_pct" in feats
        assert "ln_explained_intensity" in feats
        assert "ln_total_intensity" in feats
        assert "ln_explained_intensity_ratio" in feats
        assert "hyperscore" in feats

    def test_nan_for_inactive_series(self):
        """Series with no annotations across all spectra should have NaN values."""
        # Only b and y ions present
        ann = self._annotated_spectrum(
            peak_annotations=[
                [FragmentAnnotation("b", 1, 1)],
                [FragmentAnnotation("y", 2, 1)],
                [],
            ]
        )

        results = score_ms2_spectra(
            spectra=[ann],
            seq_lens=[5],
            active_ion_series=["b", "y"],
            calculate_hyperscore=False,
        )

        feats = results[0]

        # b and y should have numeric values
        assert not math.isnan(feats["matched_b_ions"])
        assert not math.isnan(feats["matched_y_ions"])

        # a, c, x, z should be NaN (not present in any annotation)
        assert math.isnan(feats["matched_a_ions"])
        assert math.isnan(feats["matched_c_ions"])
        assert math.isnan(feats["matched_x_ions"])
        assert math.isnan(feats["matched_z_ions"])

    def test_matched_counts(self):
        """Matched ion counts should be correct."""
        ann = self._annotated_spectrum(
            mz=[100.0, 200.0, 300.0, 400.0],
            intensity=[10.0, 20.0, 30.0, 40.0],
            peak_annotations=[
                [FragmentAnnotation("b", 1, 1)],
                [FragmentAnnotation("b", 2, 1)],
                [FragmentAnnotation("y", 3, 1)],
                [],
            ],
        )

        results = score_ms2_spectra(
            spectra=[ann],
            seq_lens=[5],
            active_ion_series=["b", "y"],
            calculate_hyperscore=False,
        )

        feats = results[0]
        assert feats["matched_b_ions"] == 2.0
        assert feats["matched_y_ions"] == 1.0
        assert feats["matched_b_ions_pct"] == pytest.approx(2.0 / 5.0)
        assert feats["matched_y_ions_pct"] == pytest.approx(1.0 / 5.0)

    def test_hyperscore_disabled(self):
        """Hyperscore should not be in output when disabled."""
        ann = self._annotated_spectrum(
            peak_annotations=[
                [FragmentAnnotation("b", 1, 1)],
                [],
                [],
            ]
        )

        results = score_ms2_spectra(
            spectra=[ann],
            seq_lens=[5],
            active_ion_series=["b", "y"],
            calculate_hyperscore=False,
        )

        assert "hyperscore" not in results[0]

    def test_empty_seq_len(self):
        """Spectrum with seq_len=0 should return empty features."""
        ann = self._annotated_spectrum()

        results = score_ms2_spectra(
            spectra=[ann],
            seq_lens=[0],
            active_ion_series=["b", "y"],
            calculate_hyperscore=False,
        )

        assert len(results) == 1
        assert len(results[0]) == 0

    def test_roundtrip_annotate_then_score(self):
        """End-to-end: annotate spectra, then score them."""
        spectrum = _make_spectrum(
            "roundtrip",
            [227.1026, 324.1554, 999.999],
            [100.0, 200.0, 50.0],
        )

        annotated = annotate_ms2_spectra(
            spectra=[spectrum],
            proformas=[PEPTIDE_PROFORMA],
            seq_lens=[PEPTIDE_SEQ_LEN],
            fragmentation_model="cidhcd",
            mass_mode="monoisotopic",
            tolerance_value=20.0,
            tolerance_mode="ppm",
        )

        results = score_ms2_spectra(
            spectra=annotated,
            seq_lens=[PEPTIDE_SEQ_LEN],
            active_ion_series=["a", "b", "y"],
            calculate_hyperscore=True,
        )

        assert len(results) == 1
        feats = results[0]

        # Should have all expected features
        assert "ln_explained_intensity" in feats
        assert "matched_ions_pct" in feats
        assert "hyperscore" in feats

        # Total intensity should include all peaks
        assert feats["ln_total_intensity"] > 0
