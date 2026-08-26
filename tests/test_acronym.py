from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import acronym


def make_record(tmp_path: Path, name: str, ra: str, dec: str = "+00:00:00") -> acronym.FrameRecord:
    header = fits.Header()
    header["DSEC11"] = "[1:2,1:2]"
    return acronym.FrameRecord(
        path=tmp_path / name,
        objtype="Object",
        filt="r",
        exposure=10.0,
        objname="test",
        readamps="Quad",
        binning=(2, 2),
        ra=ra,
        dec=dec,
        header=header,
    )


def quad_header(imagetype: str, filt: str, exposure: float) -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = imagetype
    header["FILTER"] = filt
    header["EXPTIME"] = exposure
    header["READAMPS"] = "Quad"
    header["CCDBIN1"] = 2
    header["CCDBIN2"] = 2
    header["DSEC11"] = "[1:8,1:8]"
    header["BSEC11"] = "[9:10,1:8]"
    header["DSEC21"] = "[11:18,1:8]"
    header["BSEC21"] = "[19:20,1:8]"
    header["DSEC12"] = "[1:8,9:16]"
    header["BSEC12"] = "[9:10,9:16]"
    header["DSEC22"] = "[11:18,9:16]"
    header["BSEC22"] = "[19:20,9:16]"
    return header


def write_quad(
    path: Path,
    imagetype: str,
    filt: str,
    exposure: float,
    active_signal: float,
    ra: str | None = None,
    drop_ll: bool = False,
) -> None:
    overscan = 100.0
    data = np.full((16, 20), overscan, dtype=np.float32)
    for yslice, xslice in (
        (slice(0, 8), slice(0, 8)),
        (slice(0, 8), slice(10, 18)),
        (slice(8, 16), slice(0, 8)),
        (slice(8, 16), slice(10, 18)),
    ):
        data[yslice, xslice] = overscan + active_signal
    if drop_ll:
        data[0:8, 0:8] = 65_310.0
        data[0:8, 8:10] = 65_310.0
    header = quad_header(imagetype, filt, exposure)
    if imagetype == "Object":
        header["OBJNAME"] = "synthetic"
        header["RA"] = ra or "00:00:00"
        header["DEC"] = "+00:00:00"
    fits.writeto(path, data, header=header)


def make_synthetic_dataset(
    tmp_path: Path, science_count: int = 4, drop_ll: bool = False
) -> Path:
    write_quad(
        tmp_path / "bias.0001.fits", "Bias", "CU VR", 0.0, 5.0, drop_ll=drop_ll
    )
    write_quad(
        tmp_path / "bias.0002.fits", "Bias", "CU VR", 0.0, 5.0, drop_ll=drop_ll
    )
    write_quad(
        tmp_path / "dark.0001.fits", "Dark", "CU VR", 10.0, 7.0, drop_ll=drop_ll
    )
    write_quad(
        tmp_path / "dark.0002.fits", "Dark", "CU VR", 10.0, 7.0, drop_ll=drop_ll
    )
    write_quad(
        tmp_path / "flat.0001.fits",
        "Flat",
        "SDSS r #1",
        10.0,
        10007.0,
        drop_ll=drop_ll,
    )
    # A different lamp level verifies per-frame normalization before combination.
    write_quad(
        tmp_path / "flat.0002.fits",
        "Flat",
        "SDSS r #1",
        10.0,
        20007.0,
        drop_ll=drop_ll,
    )
    for index in range(science_count):
        # Thirty arcseconds between pointings at the equator.
        seconds = index * 2
        write_quad(
            tmp_path / f"image.{index + 1:04d}.fits",
            "Object",
            "SDSS r #1",
            10.0,
            1007.0,
            ra=f"00:00:{seconds:02d}",
            drop_ll=drop_ll,
        )
    return tmp_path


def write_ur(path: Path, imagetype: str = "Object") -> None:
    data = np.full((8, 10), 100.0, dtype=np.float32)
    data[:, :8] = 125.0
    header = fits.Header()
    header["IMAGETYP"] = imagetype
    header["FILTER"] = "SDSS r #1"
    header["EXPTIME"] = 10.0
    header["READAMPS"] = "UR"
    header["CCDBIN1"] = 2
    header["CCDBIN2"] = 2
    header["DSEC22"] = "[1:8,1:8]"
    header["BSEC22"] = "[9:10,1:8]"
    if imagetype == "Object":
        header["OBJNAME"] = "synthetic"
        header["RA"] = "00:00:00"
        header["DEC"] = "+00:00:00"
    fits.writeto(path, data, header=header)


def test_filter_normalization_preserves_multi_letter_filter() -> None:
    assert acronym.normalize_filter("SDSS r #1") == "r"
    assert acronym.normalize_filter("CU VR") == "CU VR"
    assert acronym.clean_filtname("CU VR") == "CUVR"


def test_trim_image_quad_subtracts_each_amplifier_overscan(tmp_path: Path) -> None:
    header = quad_header("Object", "r", 10.0)
    data = np.full((16, 20), 100, dtype=np.uint16)
    data[0:8, 0:8] = 110
    data[0:8, 10:18] = 120
    data[8:16, 0:8] = 130
    data[8:16, 10:18] = 140
    path = tmp_path / "quad.fits"
    fits.writeto(path, data, header=header)

    trimmed, result_header = acronym.trim_image(path)

    assert trimmed.shape == (16, 16)
    np.testing.assert_allclose(trimmed[:8, :8], 10.0, atol=1e-5)
    np.testing.assert_allclose(trimmed[:8, 8:], 20.0, atol=1e-5)
    np.testing.assert_allclose(trimmed[8:, :8], 30.0, atol=1e-5)
    np.testing.assert_allclose(trimmed[8:, 8:], 40.0, atol=1e-5)
    assert result_header["OVSCORR"]


def test_trim_image_auto_nulls_saturated_quad_and_allows_override(
    tmp_path: Path,
) -> None:
    header = quad_header("Object", "r", 10.0)
    data = np.full((16, 20), 100.0, dtype=np.float32)
    data[0:8, 0:8] = 65_310.0
    data[0:8, 8:10] = 65_310.0
    data[0:8, 10:18] = 120.0
    data[8:16, 0:8] = 130.0
    data[8:16, 10:18] = 140.0
    path = tmp_path / "dropped-ll.fits"
    fits.writeto(path, data, header=header)

    trimmed, result_header = acronym.trim_image(path)

    assert np.all(np.isnan(trimmed[:8, :8]))
    np.testing.assert_allclose(trimmed[:8, 8:], 20.0, atol=1e-5)
    np.testing.assert_allclose(trimmed[8:, :8], 30.0, atol=1e-5)
    np.testing.assert_allclose(trimmed[8:, 8:], 40.0, atol=1e-5)
    assert result_header["BADAMPS"] == "LL"

    unmasked, unmasked_header = acronym.trim_image(path, bad_amp="none")
    assert np.all(np.isfinite(unmasked))
    assert "BADAMPS" not in unmasked_header

    healthy_path = tmp_path / "healthy.fits"
    write_quad(healthy_path, "Object", "r", 10.0, 25.0)
    forced, forced_header = acronym.trim_image(healthy_path, bad_amp="LL")
    assert np.all(np.isnan(forced[:8, :8]))
    assert np.all(np.isfinite(forced[:8, 8:]))
    assert forced_header["BADAMPS"] == "LL"


def test_trim_image_ur_uses_full_frame_amp_22_sections(tmp_path: Path) -> None:
    path = tmp_path / "ur.fits"
    write_ur(path)

    trimmed, header = acronym.trim_image(path)

    assert trimmed.shape == (8, 8)
    np.testing.assert_allclose(trimmed, 25.0, atol=1e-5)
    assert header["READAMPS"] == "UR"
    assert header["OVSCORR"]
    assert "BADAMPS" not in header


def test_catalog_skips_truncated_fits_payload(tmp_path: Path) -> None:
    write_quad(tmp_path / "good.fits", "Bias", "r", 0.0, 5.0)
    truncated = tmp_path / "truncated.fits"
    write_quad(truncated, "Flat", "r", 1.0, 1000.0)
    truncated.write_bytes(truncated.read_bytes()[:3500])

    records, failures = acronym.catalog_frames(tmp_path)

    assert [record.path.name for record in records] == ["good.fits"]
    assert len(failures) == 1
    assert failures[0][0].name == "truncated.fits"


def test_mixed_quad_and_ur_configs_are_rejected(tmp_path: Path) -> None:
    write_quad(tmp_path / "quad.fits", "Bias", "r", 0.0, 5.0)
    write_ur(tmp_path / "ur.fits", imagetype="Bias")
    records, failures = acronym.catalog_frames(tmp_path)
    assert not failures

    with pytest.raises(ValueError, match="Multiple detector configurations"):
        acronym._validate_detector_config(records)


def test_select_dark_uses_exact_then_longest_dark() -> None:
    darks = {10.0: np.ones((2, 2)), 30.0: np.full((2, 2), 3.0)}
    exact, source = acronym.select_dark(10.0, darks)
    np.testing.assert_allclose(exact, 1.0)
    assert source == 10.0

    scaled, source = acronym.select_dark(20.0, darks)
    np.testing.assert_allclose(scaled, 2.0)
    assert source == 30.0


def test_complete_linkage_does_not_chain_boresights(tmp_path: Path) -> None:
    records = [
        make_record(tmp_path, "a.fits", "00:00:00.000"),
        make_record(tmp_path, "b.fits", "00:00:00.667"),  # ~10 arcsec
        make_record(tmp_path, "c.fits", "00:00:01.333"),  # ~20 arcsec from a
    ]
    assignments, count = acronym.cluster_boresights(records, threshold_arcsec=15.0)
    assert count == 2
    group_sizes = sorted(
        sum(assignment.group_id == group_id for assignment in assignments.values())
        for group_id in {assignment.group_id for assignment in assignments.values()}
    )
    assert group_sizes == [1, 2]
    assert len({assignment.group_id for assignment in assignments.values()}) == 2

    inside = [
        make_record(tmp_path, "inside-a.fits", "00:00:00.000"),
        make_record(tmp_path, "inside-b.fits", "00:00:00.993"),  # ~14.9 arcsec
    ]
    outside = [
        make_record(tmp_path, "outside-a.fits", "00:00:00.000"),
        make_record(tmp_path, "outside-b.fits", "00:00:01.007"),  # ~15.1 arcsec
    ]
    assert acronym.cluster_boresights(inside, threshold_arcsec=15.0)[1] == 1
    assert acronym.cluster_boresights(outside, threshold_arcsec=15.0)[1] == 2


def test_representative_selection_is_quality_ordered_and_deterministic(tmp_path: Path) -> None:
    records = [
        make_record(tmp_path, "a.fits", "00:00:00"),
        make_record(tmp_path, "b.fits", "00:00:00"),
        make_record(tmp_path, "c.fits", "00:00:00"),
    ]
    assignments = {
        record.path: acronym.BoresightAssignment(group_id=1, separation_arcsec=0.0)
        for record in records
    }
    candidates = [
        acronym.CandidateMetric(records[0], 0.90, 100.0, 2.0, 50.0, True),
        acronym.CandidateMetric(records[1], 0.95, 100.0, 4.0, 25.0, True),
        acronym.CandidateMetric(records[2], 0.95, 100.0, 2.0, 50.0, True),
    ]
    selected = acronym.choose_representatives(candidates, assignments)
    assert selected[1].record.path.name == "c.fits"


def test_source_mask_expands_bright_source() -> None:
    image = np.full((128, 128), 1000.0, dtype=np.float32)
    rng = np.random.default_rng(12)
    image += rng.normal(0.0, 3.0, image.shape).astype(np.float32)
    image[60:65, 60:65] += 500.0
    mask, median, noise = acronym.make_source_mask(image, sigma_threshold=3.0, dilation=5)
    assert mask[62, 62]
    assert np.count_nonzero(mask[55:70, 55:70]) > 25
    assert 990.0 < median < 1010.0
    assert noise > 0


def test_hybrid_preserves_lamp_structure_and_smooths_residual() -> None:
    yy, xx = np.indices((64, 64))
    lamp = np.where((xx + yy) % 2 == 0, 0.95, 1.05).astype(np.float32)
    gradient = (0.85 + 0.30 * xx / 63.0).astype(np.float32)
    raw = [lamp * gradient for _ in range(5)]
    residual = [gradient.copy() for _ in range(5)]

    science, illumination, hybrid, coverage = acronym.combine_superflat_inputs(
        raw, residual, lamp, smoothing_sigma=4.0
    )

    assert np.isclose(np.nanmedian(science), 1.0)
    assert np.isclose(np.nanmedian(illumination), 1.0)
    assert np.isclose(np.nanmedian(hybrid), 1.0)
    assert np.all(coverage == 5)
    assert np.nanstd(hybrid[:, 1:] - hybrid[:, :-1]) > np.nanstd(
        illumination[:, 1:] - illumination[:, :-1]
    )


def test_legacy_lamp_run_corrects_bias_and_writes_qa(tmp_path: Path) -> None:
    make_synthetic_dataset(tmp_path, science_count=1)
    (tmp_path / "image.corrupt.fits").write_bytes(b"")
    output = tmp_path / "reduced_lamp"

    result = acronym.run_pipeline(tmp_path, output_directory=output)

    assert result["science_count"] == 1
    assert len(result["failures"]) == 1
    reduced = fits.getdata(output / "data" / "red_image.0001.fits")
    np.testing.assert_allclose(reduced, 1000.0, atol=1e-3)
    header = fits.getheader(output / "data" / "red_image.0001.fits")
    assert header["BIASCOR"]
    assert header["FLATMODE"] == "lamp"
    assert (output / "cals" / "master_flat_r.fits").exists()
    assert (output / "cals" / "calibration_qa.csv").exists()


def test_dropped_amp_provenance_reaches_masters_and_science(tmp_path: Path) -> None:
    make_synthetic_dataset(tmp_path, science_count=1, drop_ll=True)
    output = tmp_path / "reduced_dropped_ll"

    result = acronym.run_pipeline(tmp_path, output_directory=output)

    assert result["bad_amplifiers"] == "LL"
    product_paths = (
        output / "cals" / "master_bias.fits",
        output / "cals" / "master_dark_10.0.fits",
        output / "cals" / "master_flat_r.fits",
        output / "data" / "red_image.0001.fits",
    )
    for path in product_paths:
        data = fits.getdata(path)
        header = fits.getheader(path)
        assert header["BADAMPS"] == "LL"
        assert "to NaN before calibration" in " ".join(header["HISTORY"])
        assert np.all(np.isnan(data[:8, :8]))
        assert np.mean(np.isfinite(data[:8, 8:])) > 0.99

    reduced = fits.getdata(output / "data" / "red_image.0001.fits")
    np.testing.assert_allclose(reduced[:8, 8:], 1000.0, atol=1e-3)


def test_superflat_products_and_sparse_fallback(tmp_path: Path) -> None:
    make_synthetic_dataset(tmp_path, science_count=4)
    hybrid_output = tmp_path / "reduced_superflat"
    result = acronym.run_pipeline(
        tmp_path,
        flat_mode="superflat",
        output_directory=hybrid_output,
        illumination_smoothing_sigma=2.0,
    )

    assert result["applied_modes"]["r"] == "hybrid"
    cals = hybrid_output / "cals"
    for name in (
        "master_lamp_flat_r.fits",
        "master_science_flat_r.fits",
        "illumination_correction_r.fits",
        "master_superflat_r.fits",
        "superflat_coverage_r.fits",
        "superflat_manifest.csv",
    ):
        assert (cals / name).exists()
    assert fits.getheader(cals / "master_superflat_r.fits")["NBORE"] == 4
    statuses = {row["status"] for row in result["manifest"]}
    assert statuses == {"selected"}

    fallback_output = tmp_path / "reduced_fallback"
    fallback = acronym.run_pipeline(
        tmp_path,
        flat_mode="superflat",
        output_directory=fallback_output,
        min_superflat_boresights=5,
        illumination_smoothing_sigma=2.0,
    )
    assert fallback["applied_modes"]["r"] == "lamp-fallback"
    assert not (fallback_output / "cals" / "master_superflat_r.fits").exists()
    assert fits.getheader(fallback_output / "cals" / "master_flat_r.fits")[
        "FLATMODE"
    ] == "lamp-fallback"
