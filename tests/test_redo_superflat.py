import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import redo_superflat as runner


def make_fits(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.ones((12, 12), dtype=np.float32))
    return path


def science_row(name="P/2025 W3", **changes):
    return {"object": name, "object_dir": "P+2025_W3", "image_type": "Object", "excluded": False,
            "source": "/raw/image.fits", "notes": "", **{stage: "success" for stage in runner.STAGES}, **changes}


def test_discovery_selects_arctic_and_refuses_ambiguity(tmp_path):
    reduced = tmp_path / "20260215UT_APO/reduced"
    make_fits(reduced / "lamp/data/red_image.0001.fits")
    master = tmp_path / "APO/Q1UW09/UT260215/arctic"
    make_fits(master / "image.0001.fits")
    make_fits(tmp_path / "APO/gcam/UT260215/image.0001.fits")
    assert runner.discover_master(reduced)[1:] == (master, "20260215")
    make_fits(tmp_path / "APO/Q1UW10/UT260215/arctic/image.0001.fits")
    with pytest.raises(runner.Pause, match="found 2"):
        runner.discover_master(reduced)
    assert runner.discover_master(reduced, master.parent)[1] == master


def test_bad_folders_are_excluded_without_reading_payload(tmp_path):
    make_fits(tmp_path / "image.fits")
    (tmp_path / "bad/nested").mkdir(parents=True)
    (tmp_path / "bad/nested/truncated.fits").write_bytes(b"")
    seen = []

    def probe(path):
        seen.append(path)
        return science_row(source=str(path))

    rows = runner.inventory(tmp_path, probe=probe)
    assert seen == [tmp_path / "image.fits"]
    assert len(rows) == 2
    assert next(r for r in rows if r["excluded"])["hydration"] == "excluded"


def test_hydration_failure_is_recorded_before_pause(tmp_path):
    (tmp_path / "empty.fits").touch()
    snapshots = []

    def fail(path):
        raise runner.Pause("Still offline")

    with pytest.raises(runner.Pause, match="Still offline"):
        runner.inventory(tmp_path, probe=fail, progress=lambda rows: snapshots.append(list(rows)))
    assert snapshots[-1][0]["hydration"] == "failed"
    assert snapshots[-1][0]["notes"] == "Still offline"


def test_bounded_hydration_retries_twice(monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("probe", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(runner.Pause, match="Hydration/read exceeded"):
        runner.bounded_probe(Path("offline.fits"), timeout=1)
    assert len(calls) == 2


def test_object_names_and_collisions():
    assert runner.object_directory("P/2025 W3") == "P+2025_W3"
    with pytest.raises(runner.Pause, match="collision"):
        runner.check_names([science_row("P/2025 W3"), science_row("P+2025 W3")])
    with pytest.raises(runner.Pause, match="Missing"):
        runner.check_names([science_row(" ")])


def test_cutout_name_uses_midpoint_band_exptime_and_thumbnail(tmp_path):
    source = tmp_path / "red_image.0045.new_chip0.fits"
    header = fits.Header()
    header["FILTER"] = "SDSS r #1"
    header["EXPTIME"] = 300.0
    fits.writeto(source, np.ones((4, 4), dtype=np.float32), header)
    row = science_row(organized_path=str(source), midpoint_jd=2461200.5)
    parts = runner.cutout_name_parts(row)
    assert parts["band"] == "r"
    assert parts["exptime"] == "300s"
    assert parts["stem"].startswith("P+2025_W3_20260609_000000_r_300s_red_image.0045.new_chip0_126arcsec_NuEl")
    assert runner.band_token("VR") == "VR"
    assert runner.band_token("r_1") == "r"


@pytest.mark.parametrize("success,total,expected", [(43, 85, True), (42, 85, False), (2, 4, False), (0, 0, False)])
def test_majority_threshold(success, total, expected):
    assert runner.majority_solved(success, total) == expected


def test_object_manifest_counts_fallbacks_and_failures(tmp_path):
    rows = [science_row(flat_mode="hybrid"),
            science_row(flat_mode="lamp-fallback", solve="failed", cutout="warning", notes="unverified WCS")]
    state = {"frames": rows, "output": str(tmp_path)}
    summary = runner.object_summary(state)[0]
    assert summary["input_count"] == 2
    assert summary["hybrid_count"] == summary["lamp_fallback_count"] == 1
    assert summary["solve_failed_count"] == summary["fallback_cutout_count"] == 1
    assert summary["solve"] == summary["cutout"] == "warning"
    runner.checkpoint(state)
    assert json.loads((tmp_path / "pipeline_state.json").read_text())["frames"] == rows
    assert "unverified WCS" in (tmp_path / "master_manifest.csv").read_text()


def test_receipts_refuse_changed_completed_artifacts(tmp_path):
    path = tmp_path / "done.txt"
    path.write_text("original")
    state = {"artifacts": {}}
    runner.receipt(state, [path])
    runner.verify_receipts(state)
    path.write_text("changed")
    with pytest.raises(runner.Pause, match="changed or missing"):
        runner.verify_receipts(state)


def test_solver_wall_timeout_kills_process(tmp_path):
    result = runner.run_bounded([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "log", .1)
    assert result == 124
    assert "WALL TIMEOUT" in (tmp_path / "log").read_text()


def linear_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [32., 32.]
    wcs.wcs.crval = [100., 20.]
    wcs.wcs.cdelt = [-.228 / 3600, .228 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def test_valid_wcs_rejects_missing_or_singular_solution():
    assert not runner.valid_wcs(fits.Header(), (64, 64))
    assert runner.valid_wcs(linear_wcs().to_header(), (64, 64))
    header = linear_wcs().to_header()
    header["CDELT1"] = 0.
    assert not runner.valid_wcs(header, (64, 64))


def test_native_cutout_and_fallback_provenance(tmp_path, monkeypatch):
    results, _ = runner.cutout_helpers()
    from PIL import Image

    def png(source, output):
        data = fits.getdata(source)
        Image.fromarray(np.zeros(data.shape, dtype=np.uint8)).save(output)

    monkeypatch.setattr(results, "coc_tools_fits2png", png)
    monkeypatch.setattr(results, "coc_tools_make_arrows", lambda a, b, out: out.write_bytes(b"%PDF-1.4\ntest"))
    monkeypatch.setattr(runner, "ephemeris", lambda *args: {"ra_deg": 100., "dec_deg": 20., "antisolar_deg": 30., "antimotion_deg": 60.})
    source = tmp_path / "P+2025_W3/ARCTIC/red_image.fits"
    source.parent.mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    header = linear_wcs().to_header()
    header["DATE-OBS"] = "2026-02-15T03:00:00"
    header["EXPTIME"] = 60.
    data = np.arange(4096, dtype=np.float32).reshape(64, 64)
    data[0, :] = np.nan
    fits.writeto(source, data, header)
    state = {"output": str(tmp_path), "artifacts": {}}
    row = science_row(organized_path=str(source), solve="failed", cutout="pending", wcs_source="original_unverified")
    runner.create_cutout(state, row)
    assert row["cutout"] == "warning"
    result = fits.getheader(row["cutout_fits"])
    assert result["CUTWCS"] == "original_unverified"
    assert result["CUTNAX"] == 551
    assert fits.getdata(row["cutout_fits"]).shape == (551, 551)
    assert result["CUTRSMP"] is False
    assert result["ASTSTAT"] == "UNVERIFIED"
    assert np.array_equal(fits.getdata(source), data, equal_nan=True)
    runner.verify_receipts(state)


def test_missing_wcs_skips_cutouts_but_records_reason(tmp_path):
    row = science_row(solve="failed", cutout="pending", wcs_source="unavailable")
    state = {"frames": [row], "output": str(tmp_path)}
    runner.cutout_frames(state)
    assert row["cutout"] == "skipped"
    assert "without usable WCS" in row["notes"]


def test_make_gifs_preserves_png_dimensions_and_uses_250ms(tmp_path):
    from PIL import Image, ImageSequence

    object_root = tmp_path / "P+2025_W3" / "ARCTIC" / "cutouts"
    object_root.mkdir(parents=True)
    pngs = []
    for index, value in enumerate((20, 40, 60)):
        path = object_root / f"frame{index}.png"
        Image.new("RGBA", (17, 13), (value, value, value, 255)).save(path)
        pngs.append(path)
    rows = [science_row(cutout="success", cutout_png=str(path), midpoint_jd=2460000.0 + index,
                        object_dir="P+2025_W3", gif="pending")
            for index, path in enumerate(pngs)]
    state = {"output": str(tmp_path), "frames": rows, "artifacts": {}}
    runner.make_gifs(state)
    outputs = list((tmp_path / "P+2025_W3").glob("*.gif"))
    assert len(outputs) == 1
    output = outputs[0]
    assert output.name.startswith("P+2025_W3_")
    assert output.is_file()
    with Image.open(output) as gif:
        frames = list(ImageSequence.Iterator(gif))
        assert gif.size == (17, 13)
        assert len(frames) == 3
        assert all(frame.info["duration"] == 250 for frame in frames)
    assert all(row["gif"] == "success" and row["gif_path"] == str(output) for row in rows)
    runner.verify_receipts(state)


def test_make_gifs_skips_single_png(tmp_path):
    from PIL import Image

    png = tmp_path / "single.png"
    Image.new("RGBA", (11, 11), (0, 0, 0, 255)).save(png)
    row = science_row(cutout="success", cutout_png=str(png), midpoint_jd=1.0,
                      gif="pending", object_dir="single")
    state = {"output": str(tmp_path), "frames": [row], "artifacts": {}}
    runner.make_gifs(state)
    assert row["gif"] == "skipped"
    assert not (tmp_path / "single" / "single.gif").exists()


def test_completed_stages_are_not_repeated(tmp_path, monkeypatch):
    row = science_row(solve="success", cutout="skipped")
    state = {"frames": [row], "output": str(tmp_path), "reduction_complete": True}
    runner.reduce_and_organize(state)
    runner.solve_frames(state)
    runner.cutout_frames(state)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("produce_solution", [False, True])
def test_solve_requires_marker_and_wcs_and_preserves_pixels(tmp_path, monkeypatch, produce_solution):
    source = tmp_path / "P+2025_W3/ARCTIC/red_image.fits"
    source.parent.mkdir(parents=True)
    data = np.arange(4096, dtype=np.float32).reshape(64, 64)
    data[0, 0] = np.nan
    fits.writeto(source, data, linear_wcs().to_header())
    row = science_row(solve="pending", cutout="pending", organized_path=str(source))
    state = {"output": str(tmp_path), "frames": [row], "artifacts": {},
             "solve_field": "/fake/solve-field", "astrometry_config": "/fake/config"}
    calls = []

    def solver(command, log, timeout):
        calls.append(command)
        if produce_solution:
            directory = Path(command[command.index("--dir") + 1])
            fits.writeto(directory / "solution.wcs", np.zeros((1, 1)), linear_wcs().to_header())
            (directory / "solution.solved").write_bytes(b"\x01")
        return 0

    monkeypatch.setattr(runner, "run_bounded", solver)
    if produce_solution:
        runner.solve_frames(state)
        assert row["solve"] == "success"
        assert fits.getheader(source)["ASTSTAT"] == "SOLVED"
        assert len(calls) == 1
    else:
        with pytest.raises(runner.Pause, match="majority rule"):
            runner.solve_frames(state)
        assert row["solve"] == "failed"
        assert row["wcs_source"] == "original_unverified"
        assert len(calls) == 2
    assert np.array_equal(fits.getdata(source), data, equal_nan=True)


def test_interrupted_preflight_can_resume(tmp_path, monkeypatch):
    reduced = tmp_path / "20260215UT_APO/reduced"
    make_fits(reduced / "red_image.fits")
    master = tmp_path / "raw"
    master.mkdir()
    output = reduced.parent / "superflat_processed"
    monkeypatch.setattr(runner, "discover_master", lambda *a: (reduced, master, "20260215"))

    def fail(*args, **kwargs):
        raise runner.Pause("Offline input")

    monkeypatch.setattr(runner, "preflight", fail)
    assert runner.main([str(reduced)]) == 2
    state = json.loads((output / "pipeline_state.json").read_text())
    assert state["status"] == "paused"
    assert not (output / ".pipeline.lock").exists()
    monkeypatch.setattr(runner, "preflight", lambda *args, **kwargs: {"frames": []})
    for function in ("reduce_and_organize", "solve_frames", "cutout_frames"):
        monkeypatch.setattr(runner, function, lambda state: None)
    monkeypatch.setattr(runner, "audit", lambda state: {"test": "passed"})
    assert runner.main([str(reduced), "--resume"]) == 0
    assert json.loads((output / "pipeline_state.json").read_text())["status"] == "complete"
    assert not (output / ".pipeline.lock").exists()
