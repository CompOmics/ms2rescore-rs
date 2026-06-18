import pytest
from ms2rescore_rs import get_precursor_info, get_ms2_spectra


def test_get_precursor_info_mgf():
    precursor = get_precursor_info("tests/data/test.mgf")["peptide1"]
    assert precursor.mz == pytest.approx(475.137295, 0.0001)
    assert precursor.charge == 2
    assert precursor.intensity == 0
    assert precursor.rt == pytest.approx(0.853, 0.001)
    assert precursor.im == 42.42


def test_get_precursor_info_mzml():
    precursor = get_precursor_info("tests/data/test.mzML")["index=3"]
    assert precursor.mz == pytest.approx(1007.8454316970522, 0.0001)
    assert precursor.charge == 3
    assert precursor.intensity == 0
    assert precursor.rt == pytest.approx(40.01385810926034, 0.001)
    assert precursor.im == 1.2507906843214256


def test_get_precursor_info_bruker_tdf():
    precursor = get_precursor_info("tests/data/dda_test.d")["2"]
    assert precursor.mz == pytest.approx(502, 0.0001)
    assert precursor.charge == 2
    assert precursor.intensity == 10
    assert precursor.rt == pytest.approx(0.4, 0.001)
    assert precursor.im == 1.25


def test_get_precursor_info_bruker_minitdf():
    precursor = get_precursor_info("tests/data/test.ms2")["3"]
    assert precursor.mz == pytest.approx(502, 0.0001)
    assert precursor.charge == 2
    assert precursor.intensity == 0
    assert precursor.rt == pytest.approx(0.3, 0.001)
    assert precursor.im == 1.3


def test_get_ms2_spectra_mgf():
    spectra = get_ms2_spectra("tests/data/test.mgf")
    assert len(spectra) == 1
    spectrum = spectra[0]
    assert spectrum.identifier == "peptide1"
    assert spectrum.mz[0] == pytest.approx(72.04439, 0.0001)
    assert spectrum.intensity[0] == pytest.approx(100.0, 0.0001)
    assert spectrum.mz[-1] == pytest.approx(423.11802, 0.0001)
    assert spectrum.intensity[-1] == pytest.approx(200.0, 0.0001)


def test_get_ms2_spectra_mzml():
    spectra = get_ms2_spectra("tests/data/test.mzML")
    assert len(spectra) == 4
    spectrum = spectra[0]
    assert spectrum.identifier == "index=2"
    assert spectrum.mz[0] == pytest.approx(113.69136810302734, 0.0001)
    assert spectrum.intensity[0] == pytest.approx(14.0, 0.0001)
    assert spectrum.mz[-1] == pytest.approx(1699.74951171875, 0.0001)
    assert spectrum.intensity[-1] == pytest.approx(15.0, 0.0001)


def test_get_ms2_spectra_bruker_tdf():
    spectra = get_ms2_spectra("tests/data/dda_test.d")
    assert len(spectra) == 3
    spectrum = spectra[0]
    assert spectrum.identifier == "0"
    assert spectrum.mz[0] == pytest.approx(199.7633514404297, 0.0001)
    assert spectrum.intensity[0] == pytest.approx(162.0, 0.0001)


def test_get_ms2_spectra_bruker_minitdf():
    spectra = get_ms2_spectra("tests/data/test.ms2")
    assert len(spectra) == 3
    spectrum = spectra[0]
    assert spectrum.identifier == "1"
    assert spectrum.mz[0] == pytest.approx(190.1070556640625, 0.0001)
    assert spectrum.intensity[0] == pytest.approx(350.0, 0.0001)


def test_unsupported_file_type_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_ms2_spectra("tests/data/test.unsupported")
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_precursor_info("tests/data/test.unsupported")
