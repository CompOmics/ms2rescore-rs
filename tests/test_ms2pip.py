"""Tests for ms2pip_compute_features and ms2pip_compute_theoretical_mz."""

import numpy as np
import pytest
from ms2rescore_rs import ms2pip_compute_features, ms2pip_compute_theoretical_mz

N_FEATURES = 139


class TestMs2pipComputeFeatures:
    """Tests for ms2pip_compute_features."""

    def test_basic_shape(self):
        """ACDE/2 has 4 AAs -> 3 ions -> flat array of 3*139."""
        features = ms2pip_compute_features(["ACDE/2"])
        assert len(features) == 1
        arr = features[0]
        assert arr.dtype == np.float32
        assert arr.shape == (3 * N_FEATURES,)

    def test_reshape(self):
        """Flat array reshapes to (n_ions, 139)."""
        features = ms2pip_compute_features(["PEPTIDE/2"])
        arr = features[0].reshape(-1, N_FEATURES)
        assert arr.shape == (6, N_FEATURES)  # 7 AAs -> 6 ions

    def test_peptide_level_features(self):
        """Check shared peptide-level features."""
        features = ms2pip_compute_features(["ACDE/2"])
        arr = features[0].reshape(-1, N_FEATURES)

        # p_length, p_charge
        assert arr[0, 0] == 4.0
        assert arr[0, 1] == 2.0

        # Charge one-hot: charge=2 -> [0, 1, 0, 0, 0]
        np.testing.assert_array_equal(arr[0, 2:7], [0, 1, 0, 0, 0])

        # Same shared features for all ions
        for ion in range(arr.shape[0]):
            np.testing.assert_array_equal(arr[ion, :27], arr[0, :27])

    def test_charge_onehot(self):
        """Verify charge one-hot encoding for different charges."""
        for charge, expected in [
            (1, [1, 0, 0, 0, 0]),
            (2, [0, 1, 0, 0, 0]),
            (3, [0, 0, 1, 0, 0]),
            (4, [0, 0, 0, 1, 0]),
            (5, [0, 0, 0, 0, 1]),
            (6, [0, 0, 0, 0, 1]),  # >= 5
        ]:
            features = ms2pip_compute_features([f"ACDE/{charge}"])
            arr = features[0].reshape(-1, N_FEATURES)
            np.testing.assert_array_equal(
                arr[0, 2:7], expected, err_msg=f"Failed for charge={charge}"
            )

    def test_ion_lengths(self):
        """Check n_length and c_length per ion (C code: n=i+1, c=peplen-i)."""
        features = ms2pip_compute_features(["ACDEF/2"])
        arr = features[0].reshape(-1, N_FEATURES)

        # Ion 0: n_len=1, c_len=5
        assert arr[0, 27] == 1.0
        assert arr[0, 28] == 5.0

        # Ion 1: n_len=2, c_len=4
        assert arr[1, 27] == 2.0
        assert arr[1, 28] == 4.0

        # Ion 3: n_len=4, c_len=2
        assert arr[3, 27] == 4.0
        assert arr[3, 28] == 2.0

    def test_aa_counts(self):
        """AA counts should sum to peptide length for each ion."""
        features = ms2pip_compute_features(["ACDE/2"])
        arr = features[0].reshape(-1, N_FEATURES)

        for ion in range(3):
            n_counts = arr[ion, 29:67:2]  # n_count for each AA (even indices)
            c_counts = arr[ion, 30:68:2]  # c_count for each AA (odd indices)
            assert n_counts.sum() + c_counts.sum() == 4.0  # total = peplen

    def test_batch(self):
        """Multiple peptides in batch."""
        features = ms2pip_compute_features(["ACDE/2", "PEPTIDE/3", "AK/1"])
        assert len(features) == 3
        assert features[0].shape == (3 * N_FEATURES,)
        assert features[1].shape == (6 * N_FEATURES,)
        assert features[2].shape == (1 * N_FEATURES,)

    def test_leucine_isoleucine_equivalence(self):
        """L should be treated identically to I."""
        feat_l = ms2pip_compute_features(["ACLDE/2"])
        feat_i = ms2pip_compute_features(["ACIDE/2"])
        np.testing.assert_array_equal(feat_l[0], feat_i[0])

    def test_missing_charge_raises(self):
        """ProForma without charge should raise an error."""
        with pytest.raises(Exception, match="charge"):
            ms2pip_compute_features(["ACDE"])

    def test_invalid_proforma_raises(self):
        """Invalid ProForma should raise an error."""
        with pytest.raises(Exception):
            ms2pip_compute_features(["NOT_A_PEPTIDE!!!"])

    def test_features_non_negative(self):
        """All features should be non-negative (counts, properties are all >= 0)."""
        features = ms2pip_compute_features(["PEPTIDE/2"])
        assert np.all(features[0] >= 0)

    def test_modification_handling(self):
        """Modified peptide should still produce features."""
        features = ms2pip_compute_features(["AC[Carbamidomethyl]DE/2"])
        arr = features[0].reshape(-1, N_FEATURES)
        assert arr.shape == (3, N_FEATURES)
        # Modified C should be mapped to base AA C for property lookup
        assert arr[0, 0] == 4.0  # p_length still 4


class TestMs2pipComputeTheoreticalMz:
    """Tests for ms2pip_compute_theoretical_mz."""

    def test_basic_by_ions(self):
        """ACDE/2 should produce 3 b-ions and 3 y-ions."""
        result = ms2pip_compute_theoretical_mz(
            ["ACDE/2"], ["b", "y"], "cidhcd", "monoisotopic"
        )
        assert len(result) == 1
        assert "b" in result[0]
        assert "y" in result[0]
        assert len(result[0]["b"]) == 3
        assert len(result[0]["y"]) == 3

    def test_mz_values_approximate(self):
        """Compare with ms2pip's known values for ACDE/2 (within tolerance)."""
        result = ms2pip_compute_theoretical_mz(
            ["ACDE/2"], ["b", "y"], "cidhcd", "monoisotopic"
        )
        # ms2pip C code values: b=[72.04435, 175.05354, 290.08047]
        # rustyms may differ slightly in precision
        np.testing.assert_allclose(
            result[0]["b"], [72.04435, 175.05354, 290.08047], atol=0.001
        )
        np.testing.assert_allclose(
            result[0]["y"], [148.0604, 263.0873, 366.0965], atol=0.001
        )

    def test_mz_monotonically_increasing(self):
        """m/z values should increase with ion position."""
        result = ms2pip_compute_theoretical_mz(
            ["PEPTIDE/2"], ["b", "y"], "cidhcd", "monoisotopic"
        )
        b_mz = result[0]["b"]
        y_mz = result[0]["y"]
        # Filter out zeros (unmatched positions)
        b_nonzero = [v for v in b_mz if v > 0]
        y_nonzero = [v for v in y_mz if v > 0]
        assert b_nonzero == sorted(b_nonzero)
        assert y_nonzero == sorted(y_nonzero)

    def test_batch(self):
        """Multiple peptides in batch."""
        result = ms2pip_compute_theoretical_mz(
            ["ACDE/2", "PEPTIDE/3"], ["b", "y"], "cidhcd", "monoisotopic"
        )
        assert len(result) == 2
        assert len(result[0]["b"]) == 3   # ACDE: 3 ions
        assert len(result[1]["b"]) == 6   # PEPTIDE: 6 ions

    def test_etd_ion_types(self):
        """ETD should produce c and z ions."""
        result = ms2pip_compute_theoretical_mz(
            ["PEPTIDE/2"], ["c", "z"], "etd", "monoisotopic"
        )
        assert "c" in result[0]
        assert "z" in result[0]
        # c and z ions should have some non-zero values
        assert any(v > 0 for v in result[0]["c"])
        assert any(v > 0 for v in result[0]["z"])

    def test_unrequested_ion_type_absent(self):
        """Only requested ion types should be in the output."""
        result = ms2pip_compute_theoretical_mz(
            ["ACDE/2"], ["b"], "cidhcd", "monoisotopic"
        )
        assert "b" in result[0]
        assert "y" not in result[0]

    def test_no_charge_raises(self):
        """Without a charge state, should raise (parity with feature computation)."""
        with pytest.raises(ValueError, match="No charge state"):
            ms2pip_compute_theoretical_mz(
                ["ACDE"], ["b", "y"], "cidhcd", "monoisotopic"
            )

    def test_invalid_proforma_raises(self):
        """Invalid ProForma should raise."""
        with pytest.raises(Exception):
            ms2pip_compute_theoretical_mz(
                ["NOT_A_PEPTIDE!!!"], ["b", "y"], "cidhcd", "monoisotopic"
            )
