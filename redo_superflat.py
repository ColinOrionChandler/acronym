#!/usr/bin/env python3
"""Resumable APO/ARCTIC superflat reduction, local astrometry, and cutouts."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parent
PP = Path.home() / "GitHub/photometrypipeline"
COC = Path.home() / "GitHub/coc_tools"
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
STAGES = ("hydration", "reduction", "organization", "solve", "cutout", "gif")
GIF_DURATION_MS = 250
CUTOUT_SIZE_ARCSEC = 126.0
CUTOUT_PIXELS = 551


class Pause(RuntimeError):
    """A condition requiring diagnosis or a user decision, not silent skipping."""


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def is_bad(path, root):
    return any(part.casefold() == "bad" for part in path.relative_to(root).parts[:-1])


def fits_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in FITS_SUFFIXES)


def object_directory(name):
    value = name.strip().replace("/", "+").replace(" ", "_")
    if not value or value in {".", ".."} or any(ord(c) < 32 for c in value) or "\\" in value:
        raise Pause(f"Missing or unsafe object name: {name!r}")
    if value in {"cals", "data", "logs", "astrometry", "ephemerides"}:
        raise Pause(f"Object name conflicts with pipeline directory: {name!r}")
    return value


def check_names(rows):
    names = {}
    for row in rows:
        if row.get("image_type") != "Object" or row.get("excluded"):
            continue
        name = row["object"]
        directory = object_directory(name)
        if directory in names and names[directory] != name:
            raise Pause(f"Object naming collision: {names[directory]!r} and {name!r}")
        names[directory] = name
        row["object_dir"] = directory


def discover_master(reduced, master=None):
    reduced = Path(reduced).expanduser().resolve()
    if not reduced.is_dir():
        raise Pause(f"Reduced directory does not exist: {reduced}")
    match = next((re.search(r"(20\d{6})UT_APO", p.name) for p in (reduced, *reduced.parents)
                  if re.search(r"(20\d{6})UT_APO", p.name)), None)
    if match is None:
        raise Pause("Cannot infer YYYYMMDDUT_APO observing date from the reduced path")
    date = match[1]
    datetime.strptime(date, "%Y%m%d")
    night = next(p for p in (reduced, *reduced.parents) if date + "UT_APO" in p.name)
    base = night.parent
    if master:
        selected = Path(master).expanduser().resolve()
        candidates = [selected / "arctic" if (selected / "arctic").is_dir() else selected]
    else:
        candidates = sorted((base / "APO").glob(f"*/UT{date[2:]}/arctic"))
    existing_names = {p.name.removeprefix("red_") for p in fits_files(reduced)
                      if not is_bad(p, reduced)}
    plausible = []
    for candidate in candidates:
        files = [p for p in fits_files(candidate) if not is_bad(p, candidate)] if candidate.is_dir() else []
        # A fresh night has no reduced FITS to use for filename matching. In
        # that case, accept a unique nonempty APO master; once reductions
        # exist, retain the overlap check so an unrelated UT master cannot be
        # selected accidentally.
        if files and (not existing_names or ({p.name for p in files} & existing_names)):
            plausible.append(candidate)
    if len(plausible) != 1:
        raise Pause(f"Expected one matching ARCTIC master; found {len(plausible)}: {plausible}")
    return reduced, plausible[0], date


def probe_file(path):
    """Executed in a killable subprocess so a Dropbox read cannot hang the run."""
    path = Path(path)
    before = path.stat()
    checksum = digest(path)  # Force every byte to become available.
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header
        data = hdus[0].data
        if data is None or data.ndim != 2 or not data.size:
            raise Pause("Missing or invalid primary FITS image payload")
        # Access the entire payload; do not use strict FITS-card verification:
        # ARCTIC calibration headers may contain nonstandard +NAN WCS cards.
        finite = int(np.isfinite(data).sum())
        if not finite:
            raise Pause("FITS image has no finite pixels")
        metadata = {"image_type": str(header.get("IMAGETYP", "")),
                    "object": str(header.get("OBJNAME", header.get("OBJECT", ""))).strip(),
                    "filter": str(header.get("FILTER", "")),
                    "instrument": str(header.get("INSTRUME", "")),
                    "date_obs": str(header.get("DATE-OBS", "")),
                    "readamps": str(header.get("READAMPS", "")),
                    "binning": [header.get("CCDBIN1"), header.get("CCDBIN2")],
                    "shape": list(data.shape)}
    after = path.stat()
    attrs = []
    if hasattr(os, "listxattr"):
        attrs = os.listxattr(path)
    elif shutil.which("xattr"):
        result = subprocess.run(["xattr", str(path)], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise Pause(f"Cannot inspect Dropbox attributes: {result.stderr.strip()}")
        attrs = result.stdout.splitlines()
    offline = bool(getattr(after, "st_flags", 0) & 0x40000000)
    placeholder = "com.dropbox.placeholder" in attrs
    if not after.st_size or offline or placeholder:
        raise Pause("Dropbox payload remains empty, offline, or a placeholder after reading")
    return {"source": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": checksum, "blocks": after.st_blocks, "offline": offline,
            "placeholder": placeholder,
            "hydrated_during_probe": before.st_blocks != after.st_blocks,
            **metadata}


def bounded_probe(path, timeout=120):
    error = ""
    for attempt in range(2):
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe-file", str(path)],
                                    capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return json.loads(result.stdout)
            error = result.stderr.strip()[-2000:]
        except subprocess.TimeoutExpired:
            error = f"Hydration/read exceeded {timeout} seconds"
    raise Pause(f"Cannot hydrate/read {path}: {error}")


def inventory(master, probe=bounded_probe, progress=None):
    rows = []
    for index, path in enumerate(fits_files(master), 1):
        excluded = is_bad(path, master)
        if excluded:
            stat = path.stat()
            row = {"source": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                   "image_type": "", "object": "", "filter": "", "excluded": True,
                   "notes": "Excluded: beneath a bad directory", "hydration": "excluded"}
        else:
            try:
                row = {**probe(path), "excluded": False, "notes": "", "hydration": "success"}
            except Exception as exc:
                rows.append({"source": str(path), "excluded": False, "image_type": "", "object": "",
                             "hydration": "failed", "notes": str(exc),
                             **{stage: "pending" for stage in STAGES[1:]}})
                if progress:
                    progress(rows)
                raise
        row.update({stage: "excluded" if excluded else "pending" for stage in STAGES[1:]})
        rows.append(row)
        if progress:
            progress(rows)
        if index % 20 == 0:
            print(f"Preflight: read {index} raw FITS files", flush=True)
    check_names(rows)
    return rows


def snapshot(root):
    return {str(p): {"sha256": digest(p), "bytes": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns} for p in fits_files(root)}


def preflight(reduced, master, date, progress=None, original_reduced=None):
    solve = shutil.which("solve-field")
    if not solve:
        raise Pause("Local solve-field is unavailable")
    config = Path(solve).resolve().parents[1] / "etc/astrometry.cfg"
    if not config.is_file():
        raise Pause(f"Astrometry configuration is unavailable: {config}")
    indexes = []
    for line in config.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if len(parts) == 2 and parts[0] == "add_path":
            directory = Path(parts[1]).expanduser()
            if not directory.is_absolute():
                directory = config.parent / directory
            indexes.extend(p for p in directory.glob("index-*.fits") if p.stat().st_size > 0)
    if not indexes:
        raise Pause("No accessible local astrometry indexes in the configured add_path directories")
    for module in [PP / "pptool_reduced_results.py", PP / "pptool_pp_cutouts.py", COC / "arrows.py"]:
        if not module.is_file():
            raise Pause(f"Required cutout helper unavailable: {module}")
    rows = inventory(master, progress=progress)
    eligible = [row for row in rows if not row["excluded"]]
    science = [row for row in eligible if row["image_type"] == "Object"]
    if not science:
        raise Pause("Master has no science exposures")
    configs = {(row["readamps"], tuple(row["binning"])) for row in eligible}
    if len(configs) != 1 or any("ARCTIC" not in row["instrument"].upper() for row in eligible):
        raise Pause(f"Master has unexpected instrument or mixed detector configurations: {configs}")
    if any(row["date_obs"][:10].replace("-", "") != date for row in science):
        raise Pause("Science exposure dates do not match the requested UT night")
    import acronym
    flat_filters = {acronym.normalize_filter(row["filter"]) for row in eligible if row["image_type"] == "Flat"}
    missing = {acronym.normalize_filter(row["filter"]) for row in science} - flat_filters
    if missing or not any(row["image_type"] == "Bias" for row in eligible):
        raise Pause(f"Missing calibration inputs; filters without flats: {missing}")
    if any(Path(row["source"]).parent != master or Path(row["source"]).suffix != ".fits" for row in eligible):
        raise Pause("Acronym expects top-level .fits inputs; this layout needs explicit staging before reduction")
    old = snapshot(reduced) if original_reduced is None else original_reduced
    expected_bytes = sum(np.prod(row["shape"]) * 4 for row in science)
    required = int(expected_bytes * 4 + 2 * 1024**3)
    free = shutil.disk_usage(reduced.parent).free
    if free < required:
        raise Pause(f"Insufficient output disk space: need {required} bytes, available {free}")
    return {"frames": rows, "original_reduced": old, "solve_field": solve,
            "astrometry_config": str(config), "astrometry_config_sha256": digest(config),
            "index_count": len(indexes), "free_bytes": free, "required_bytes": required,
            "science_count": len(science), "object_count": len({r['object'] for r in science})}


def aggregate_status(values):
    values = list(values)
    if all(v == "success" for v in values):
        return "success"
    if all(v == "pending" for v in values):
        return "pending"
    if "running" in values:
        return "running"
    if "pending" in values:
        return "partial"
    if all(v == "skipped" for v in values):
        return "skipped"
    if all(v in {"failed", "skipped"} for v in values):
        return "failed"
    return "warning"


def object_summary(state):
    rows = []
    for name in sorted({r["object"] for r in state["frames"] if r.get("image_type") == "Object" and not r["excluded"]}):
        frames = [r for r in state["frames"] if r.get("image_type") == "Object" and r["object"] == name and not r["excluded"]]
        directory = frames[0].get("object_dir", name.strip().replace("/", "+").replace(" ", "_"))
        row = {"object": name, "object_dir": directory, "input_count": len(frames)}
        row.update({stage: aggregate_status(f[stage] for f in frames) for stage in STAGES})
        row.update({stage + "_success_count": sum(f[stage] == "success" for f in frames) for stage in STAGES})
        row.update({"hybrid_count": sum(f.get("flat_mode") == "hybrid" for f in frames),
                    "lamp_fallback_count": sum(f.get("flat_mode") == "lamp-fallback" for f in frames),
                    "solve_failed_count": sum(f["solve"] == "failed" for f in frames),
                    "fallback_cutout_count": sum(f["cutout"] == "warning" for f in frames),
                    "cutout_product_count": sum(f["cutout"] in {"success", "warning"} for f in frames),
                    "gif_path": next((f.get("gif_path", "") for f in frames if f.get("gif_path")), ""),
                    "output_dir": str(Path(state["output"]) / directory / "ARCTIC"),
                    "notes": " | ".join(sorted({f.get("notes", "") for f in frames if f.get("notes")}))})
        rows.append(row)
    return rows


def checkpoint(state):
    root = Path(state["output"])
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(root / "pipeline_state.json", state)
    fields = ["source", "object", "object_dir", "image_type", "filter", "excluded", "sha256", "bytes",
              *STAGES, "flat_mode", "badamps", "reduced_path", "organized_path", "wcs_source",
              "solve_attempts", "datetime_token", "band_token", "exptime_token", "cutout_fits", "cutout_png",
              "arrow_pdf", "gif_path", "ephemeris_id", "notes"]
    atomic_csv(root / "frame_manifest.csv", state["frames"], fields)
    summary = object_summary(state)
    if summary:
        atomic_csv(root / "master_manifest.csv", summary, list(summary[0]))


def receipt(state, paths):
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise Pause(f"Missing or empty output: {path}")
        state["artifacts"][str(path)] = digest(path)


def verify_receipts(state):
    for path, expected in state["artifacts"].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise Pause(f"Completed output changed or missing: {path}")


def verify_sources(state):
    paths = fits_files(state["master"])
    if {str(p) for p in paths} != {r["source"] for r in state["frames"]}:
        raise Pause("Raw file inventory changed")
    for row in state["frames"]:
        path = Path(row["source"])
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise Pause(f"Raw source metadata changed: {path}")
        if row.get("sha256") and digest(path) != row["sha256"]:
            raise Pause(f"Raw source bytes changed: {path}")
    if snapshot(state["reduced"]) != state["original_reduced"]:
        raise Pause("The existing reduced FITS inventory or contents changed")


def science_frames(state):
    return [r for r in state["frames"] if r.get("image_type") == "Object" and not r["excluded"]]


def reduce_and_organize(state):
    root = Path(state["output"])
    frames = science_frames(state)
    if not state.get("reduction_complete"):
        for row in frames:
            row["reduction"] = "running"
        checkpoint(state)
        print(f"Reducing {len(frames)} science exposures; log: {root / 'logs/reduction.log'}", flush=True)
        import acronym
        with (root / "logs/reduction.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            result = acronym.run_pipeline(state["master"], flat_mode="superflat", output_directory=root)
        if result["failures"] or result["science_count"] != len(frames):
            raise Pause(f"Reduction did not account for every eligible science exposure: {result['failures']}")
        outputs = sorted((root / "data").glob("*.fits"))
        if len(outputs) != len(frames):
            raise Pause("Reduced exposure count differs from the science inventory")
        for row in frames:
            path = root / "data" / ("red_" + Path(row["source"]).name)
            with fits.open(path, memmap=False) as hdus:
                hdus.verify("exception")
                if not np.isfinite(hdus[0].data).any():
                    raise Pause(f"No finite reduced science pixels: {path}")
                row["flat_mode"] = hdus[0].header["FLATMODE"]
                row["badamps"] = hdus[0].header.get("BADAMPS", "")
            if row["flat_mode"] not in {"hybrid", "lamp-fallback"}:
                raise Pause(f"Unexpected applied flat mode: {row['flat_mode']}")
            row.update(reduction="success", reduced_path=str(path))
        with (root / "cals/calibration_qa.csv").open() as stream:
            qa = list(csv.DictReader(stream))
        if {q["filename"] for q in qa} != {Path(r["source"]).name for r in frames} or len(qa) != len(frames):
            raise Pause("Calibration QA does not cover each exposure exactly once")
        receipt(state, [*outputs, *(p for p in (root / "cals").iterdir() if p.is_file())])
        for row in state["frames"]:
            if not row["excluded"] and row["image_type"] != "Object":
                row.update(reduction="success", organization="not_applicable", solve="not_applicable", cutout="not_applicable", gif="not_applicable")
        state["reduction_complete"] = True
        checkpoint(state)
    for row in frames:
        if row["organization"] == "success":
            continue
        destination = root / row["object_dir"] / "ARCTIC" / Path(row["reduced_path"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["reduced_path"], destination)
        row.update(organized_path=str(destination), organization="success")
        receipt(state, [destination])
        checkpoint(state)


def valid_wcs(header, shape):
    try:
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            return False
        matrix = wcs.pixel_scale_matrix
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-16:
            return False
        x = np.array([0, (shape[1] - 1) / 2, shape[1] - 1])
        y = np.array([0, (shape[0] - 1) / 2, shape[0] - 1])
        ra, dec = wcs.pixel_to_world_values(x, y)
        xx, yy = wcs.world_to_pixel_values(ra, dec)
        return bool(np.isfinite([ra, dec, xx, yy]).all() and np.allclose(xx, x, atol=.1) and np.allclose(yy, y, atol=.1))
    except Exception:
        return False


def run_bounded(command, log_path, timeout):
    with Path(log_path).open("w") as log:
        log.write(shlex.join([str(v) for v in command]) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.write(f"\nWALL TIMEOUT after {timeout} seconds\n")
            return 124


def cutout_helpers():
    if str(PP) not in sys.path:
        sys.path.insert(0, str(PP))
    import pptool_reduced_results as results
    import pptool_pp_cutouts as cutouts
    return results, cutouts


def solve_frames(state):
    root = Path(state["output"])
    _, helpers = cutout_helpers()
    frames = science_frames(state)
    for index, row in enumerate(frames, 1):
        if row["solve"] in {"success", "failed"}:
            continue
        source = Path(row["organized_path"])
        with fits.open(source, memmap=False) as hdus:
            header, data = hdus[0].header.copy(), hdus[0].data.copy()
        usable = valid_wcs(header, data.shape)
        if usable:
            wcs = WCS(header).celestial
            ra, dec = wcs.pixel_to_world_values((data.shape[1] - 1) / 2, (data.shape[0] - 1) / 2)
            scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
        else:
            try:
                point = SkyCoord(header["RA"], header["DEC"], unit=(u.hourangle, u.deg))
                ra, dec = point.ra.deg, point.dec.deg
                scale = .114 * float(header["CCDBIN1"])
            except Exception as exc:
                raise Pause(f"Cannot establish ARCTIC solve hints for {source}: {exc}") from exc
        row["solve"] = "running"
        checkpoint(state)
        solved = False
        for attempt in (1, 2):
            directory = root / "astrometry" / source.stem / f"attempt{attempt}"
            directory.mkdir(parents=True, exist_ok=True)
            # Scratch outputs may be replaced after interruption; source FITS never are.
            (directory / "solution.solved").unlink(missing_ok=True)
            (directory / "solution.wcs").unlink(missing_ok=True)
            cpu = 60 if attempt == 1 else 120
            tolerance = .15 if attempt == 1 else .35
            command = [state["solve_field"], str(source), "--config", state["astrometry_config"],
                       "--dir", str(directory), "--out", "solution", "--overwrite", "--no-plots",
                       "--new-fits", "none", "--no-verify", "--ra", str(float(ra)), "--dec", str(float(dec)),
                       "--radius", "1" if attempt == 1 else "5", "--scale-units", "arcsecperpix",
                       "--scale-low", str(scale * (1 - tolerance)), "--scale-high", str(scale * (1 + tolerance)),
                       "--cpulimit", str(cpu), "--downsample", "2" if attempt == 1 else "1"]
            rc = run_bounded(command, directory / "solve.log", timeout=cpu + 60)
            row["solve_attempts"] = attempt
            row["solve_returncode"] = rc
            marker, wcs_path = directory / "solution.solved", directory / "solution.wcs"
            if rc == 0 and marker.is_file() and marker.read_bytes()[:1] == b"\x01" and wcs_path.is_file():
                new_wcs_header = fits.getheader(wcs_path)
                if valid_wcs(new_wcs_header, data.shape):
                    new_header = helpers._replace_celestial_wcs(header, WCS(new_wcs_header))
                    new_header["ASTSTAT"] = ("SOLVED", "Local astrometry.net solution")
                    new_header.add_history("WCS solved by local astrometry.net; pixel array unchanged")
                    candidate = source.with_suffix(".fits.tmp")
                    fits.writeto(candidate, data, new_header, overwrite=True)
                    if not np.array_equal(fits.getdata(candidate), data, equal_nan=True):
                        raise Pause(f"Astrometry changed science pixels: {source}")
                    candidate.replace(source)
                    row.update(solve="success", wcs_source="astrometry.net")
                    receipt(state, [source, wcs_path, marker])
                    solved = True
                    break
        if not solved:
            row.update(solve="failed", wcs_source="original_unverified" if usable else "unavailable")
            row["notes"] = f"Local solve failed after two attempts; WCS: {row['wcs_source']}"
        checkpoint(state)
        print(f"Astrometry {index}/{len(frames)}: {row['object']} {source.name}: {row['solve']}", flush=True)
    successes = sum(row["solve"] == "success" for row in frames)
    if not majority_solved(successes, len(frames)):
        raise Pause(f"Only {successes}/{len(frames)} exposures solved; majority rule requires more than half")


def majority_solved(successes, total):
    return total > 0 and successes * 2 > total


def ephemeris(state, row, midpoint):
    from astroquery.jplhorizons import Horizons
    target = state["settings"]["target_map"].get(row["object"], row["object"])
    cache_key = hashlib.sha256(f"{target}|705|{midpoint.jd:.12f}".encode()).hexdigest()
    path = Path(state["output"]) / "ephemerides" / (cache_key + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        if result["query_id"] != target or result["jd"] != float(midpoint.jd):
            raise Pause(f"Ephemeris cache identity mismatch: {path}")
    else:
        last = None
        for attempt in range(2):
            try:
                query = Horizons(id=target, id_type="smallbody", location="705", epochs=float(midpoint.jd))
                query.TIMEOUT = 60
                table = query.ephemerides(cache=False)  # Durable run-local JSON cache below.
                if len(table) != 1:
                    raise Pause(f"Expected one Horizons result for {target}")
                eph = table[0]
                result = {"query_id": target, "resolved_name": str(eph["targetname"]), "site": "705",
                          "jd": float(midpoint.jd), "ra_deg": float(eph["RA"]), "dec_deg": float(eph["DEC"]),
                          "antisolar_deg": float(eph["sunTargetPA"]), "antimotion_deg": float(eph["velocityPA"])}
                if not np.isfinite([result[k] for k in ("ra_deg", "dec_deg", "antisolar_deg", "antimotion_deg")]).all():
                    raise Pause(f"Invalid Horizons coordinates/vectors for {target}")
                atomic_json(path, result)
                break
            except Exception as exc:
                last = exc
                if isinstance(exc, ValueError):
                    break  # Ambiguous/unknown target requires resolution, not repeated requests.
        else:
            raise Pause(f"Horizons lookup failed for {target}: {last}")
        if not path.is_file():
            raise Pause(f"Horizons target needs resolution for {target}: {last}")
    receipt(state, [path])
    row["ephemeris_id"] = result["query_id"]
    row["ephemeris_name"] = result["resolved_name"]
    return result


def datetime_token(midpoint_jd):
    """Return a UTC exposure-midpoint token rounded to the nearest second."""
    value = Time(float(midpoint_jd), format="jd", scale="utc").to_datetime(timezone=timezone.utc)
    value = value + timedelta(microseconds=500)
    return value.replace(microsecond=0).strftime("%Y%m%d_%H%M%S")


def band_token(header_or_value):
    """Use the literal filter band, dropping only survey suffix numbering."""
    raw = str(header_or_value.get("FILTER", "") if hasattr(header_or_value, "get") else header_or_value).strip()
    value = re.sub(r"(?i)\bsdss\b", "", raw)
    value = re.sub(r"[\s_#-]*\d+$", "", value).strip()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "unknown"


def exptime_token(header_or_value):
    """Return an exposure time suitable for a stable filename component."""
    value = float(header_or_value.get("EXPTIME", 0) if hasattr(header_or_value, "get") else header_or_value)
    if not np.isfinite(value) or value <= 0:
        raise Pause(f"Missing or invalid exposure time for cutout naming: {value!r}")
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{text}s"


def thumbnail_rest(source):
    """Keep the complete reduced thumbnail identity and orientation."""
    stem = Path(source).stem
    return f"{stem}_{int(CUTOUT_SIZE_ARCSEC)}arcsec_NuEl"


def cutout_name_parts(row, midpoint_jd=None, header=None):
    source = Path(row.get("organized_path", row.get("source", "image.fits")))
    if header is None:
        header = fits.getheader(source) if source.is_file() else fits.Header()
    if midpoint_jd is None:
        midpoint_jd = row.get("midpoint_jd")
    if midpoint_jd is None and source.is_file():
        results, _ = cutout_helpers()
        midpoint_jd = float(results.exposure_midpoint(header).jd)
    if midpoint_jd is None:
        midpoint_jd = 0.0
    band = band_token(header if header.get("FILTER") else row.get("filter", ""))
    exptime = exptime_token(header if header.get("EXPTIME") else row.get("exptime", 1))
    prefix = f"{row['object_dir']}_{datetime_token(midpoint_jd)}_{band}_{exptime}"
    return {"datetime": datetime_token(midpoint_jd), "band": band, "exptime": exptime,
            "stem": f"{prefix}_{thumbnail_rest(source)}"}


def cutout_product_paths(row, midpoint_jd=None, header=None):
    source = Path(row.get("organized_path", row.get("source", "image.fits")))
    stem = cutout_name_parts(row, midpoint_jd=midpoint_jd, header=header)["stem"]
    directory = source.parent / "cutouts"
    return {"fits": directory / f"{stem}.fits", "png": directory / f"{stem}.png",
            "arrows": directory / f"{stem}_arrows.pdf"}


def gif_product_path(state, first_row):
    source = Path(first_row.get("organized_path", first_row.get("source", "image.fits")))
    header = fits.getheader(source) if source.is_file() else None
    parts = cutout_name_parts(first_row, header=header)
    return Path(state["output"]) / first_row["object_dir"] / f"{parts['stem']}.gif"


def create_cutout(state, row):
    results, helpers = cutout_helpers()
    source = Path(row["organized_path"])
    header = fits.getheader(source)
    midpoint = results.exposure_midpoint(header)
    eph = ephemeris(state, row, midpoint)
    data, wcs, source_header, info = helpers._build_cutout(
        source, eph["ra_deg"], eph["dec_deg"], CUTOUT_SIZE_ARCSEC,
        size_px=CUTOUT_PIXELS)
    if not info["inside"] or not info["overlap"]:
        if row["solve"] == "failed":
            row.update(cutout="skipped", notes=row["notes"] + "; predicted object outside usable original WCS image")
            return
        raise Pause(f"Solved image does not contain predicted target: {source}")
    try:
        data, wcs, transform = results.orient_cutout_discrete(data, wcs)
    except ValueError as exc:
        if "distortions" not in str(exc):
            raise
        transform = "identity"
    header = helpers._replace_celestial_wcs(source_header, wcs)
    for key, value in {"CUTSRC": source.name, "CUTSIZE": CUTOUT_SIZE_ARCSEC, "CUTNAX": CUTOUT_PIXELS,
                       "CUTRA": eph["ra_deg"], "CUTDEC": eph["dec_deg"],
                       "CUTIN": True, "CUTOVER": True, "CUTSCALE": info["source_pixel_scale_arcsec"],
                       "CUTMODE": "D4-PIX", "CUTRSMP": False, "CUTXFM": transform,
                       "CUTWCS": row["wcs_source"], "ASTSTAT": "SOLVED" if row["solve"] == "success" else "UNVERIFIED",
                       "MIDTIMJD": float(midpoint.jd)}.items():
        header[key] = value
    directory = source.parent / "cutouts"
    directory.mkdir(exist_ok=True)
    paths = cutout_product_paths(row, midpoint_jd=float(midpoint.jd), header=fits.getheader(source))
    output, png, arrows = paths["fits"], paths["png"], paths["arrows"]
    directory.mkdir(exist_ok=True)
    fits.writeto(output, data, header, overwrite=True)
    with (Path(state["output"]) / "logs/cutouts.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        results.coc_tools_fits2png(output, png)
        results.coc_tools_make_arrows(eph["antisolar_deg"], eph["antimotion_deg"], arrows)
    from PIL import Image
    with Image.open(png) as image:
        if image.size != (data.shape[1], data.shape[0]):
            raise Pause(f"PNG size differs from native FITS cutout: {png}")
        image.verify()
    if not arrows.read_bytes().startswith(b"%PDF"):
        raise Pause(f"Invalid arrow PDF: {arrows}")
    parts = cutout_name_parts(row, midpoint_jd=float(midpoint.jd), header=fits.getheader(source))
    row.update(cutout="success" if row["solve"] == "success" else "warning",
               cutout_fits=str(output), cutout_png=str(png), arrow_pdf=str(arrows),
               target_x=info["x"], target_y=info["y"], midpoint_jd=float(midpoint.jd),
               datetime_token=parts["datetime"], band_token=parts["band"], exptime_token=parts["exptime"])
    receipt(state, [output, png, arrows])


def rename_product(state, old, new):
    """Move a completed product and carry its receipt to the new pathname."""
    old, new = Path(old), Path(new)
    if old == new:
        return
    if old.is_file():
        if new.exists():
            raise Pause(f"Cutout naming collision: both old and new products exist: {old} and {new}")
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
    elif not new.is_file():
        return
    expected = state["artifacts"].pop(str(old), None)
    if expected is not None and digest(new) != expected:
        raise Pause(f"Renamed product changed during migration: {new}")
    state["artifacts"][str(new)] = digest(new)


def rename_products(state):
    """Migrate older cutout/GIF names without recomputing scientific products."""
    for row in science_frames(state):
        if row.get("cutout") not in {"success", "warning"}:
            continue
        old_paths = {"fits": row.get("cutout_fits"), "png": row.get("cutout_png"),
                     "arrows": row.get("arrow_pdf")}
        if not any(old_paths.values()):
            continue
        header = fits.getheader(row["organized_path"])
        desired = cutout_product_paths(row, midpoint_jd=row.get("midpoint_jd"), header=header)
        for key, old in old_paths.items():
            if old:
                rename_product(state, old, desired[key])
                row[{"fits": "cutout_fits", "png": "cutout_png", "arrows": "arrow_pdf"}[key]] = str(desired[key])
        parts = cutout_name_parts(row, midpoint_jd=row.get("midpoint_jd"), header=header)
        row.update(datetime_token=parts["datetime"], band_token=parts["band"], exptime_token=parts["exptime"])
        checkpoint(state)
    by_object = {}
    for row in science_frames(state):
        by_object.setdefault(row["object"], []).append(row)
    for rows in by_object.values():
        if not any(row.get("gif") == "success" and row.get("gif_path") for row in rows):
            continue
        first = sorted(rows, key=lambda row: (float(row.get("midpoint_jd", 0.0)), row.get("source", "")))[0]
        desired = gif_product_path(state, first)
        old = next((row.get("gif_path") for row in rows if row.get("gif_path")), None)
        if old:
            rename_product(state, old, desired)
        for row in rows:
            row["gif_path"] = str(desired)
        checkpoint(state)


def cutout_frames(state):
    frames = science_frames(state)
    for index, row in enumerate(frames, 1):
        if row["cutout"] in {"success", "warning"}:
            fits_path, png_path = row.get("cutout_fits"), row.get("cutout_png")
            if fits_path and png_path and Path(fits_path).is_file() and Path(png_path).is_file():
                with fits.open(fits_path, memmap=False) as hdus:
                    fits_shape = tuple(hdus[0].data.shape)
                from PIL import Image
                with Image.open(png_path) as image:
                    png_shape = image.size[::-1]
                if fits_shape == (CUTOUT_PIXELS, CUTOUT_PIXELS) and png_shape == (CUTOUT_PIXELS, CUTOUT_PIXELS):
                    continue
            row["cutout"] = "pending"
            row["gif"] = "pending"
        elif row["cutout"] == "skipped":
            continue
        if row["wcs_source"] == "unavailable":
            row.update(cutout="skipped", notes=row["notes"] + "; cannot center a cutout without usable WCS")
        else:
            row["cutout"] = "running"
            checkpoint(state)
            try:
                create_cutout(state, row)
            except Exception as exc:
                row["cutout"] = "failed"
                row["notes"] = (row["notes"] + "; " + str(exc)).strip("; ")
                checkpoint(state)
                raise
        checkpoint(state)
        print(f"Cutouts {index}/{len(frames)}: {row['object']}: {row['cutout']}", flush=True)


def make_gifs(state):
    """Create one 250 ms GIF per object without resizing its PNG frames."""
    from PIL import Image, ImageSequence

    by_object = {}
    for row in science_frames(state):
        by_object.setdefault(row["object"], []).append(row)
    for index, (name, rows) in enumerate(sorted(by_object.items()), 1):
        if all(row.get("gif") == "success" for row in rows):
            continue
        usable = [row for row in rows if row.get("cutout") in {"success", "warning"} and row.get("cutout_png")]
        if not usable:
            for row in rows:
                row.update(gif="skipped", notes=(row.get("notes", "") + "; no cutout PNGs for GIF").strip("; "))
            checkpoint(state)
            continue
        if len(usable) == 1:
            for row in rows:
                row.update(gif="skipped", notes=(row.get("notes", "") + "; one PNG only; GIF not created").strip("; "))
            checkpoint(state)
            print(f"GIFs {index}/{len(by_object)}: {name}: skipped (one PNG)", flush=True)
            continue
        usable.sort(key=lambda row: (float(row.get("midpoint_jd", 0.0)), row.get("source", "")))
        images = []
        dimensions = set()
        try:
            for row in usable:
                with Image.open(row["cutout_png"]) as image:
                    image.load()
                    dimensions.add(image.size)
                    images.append(image.convert("RGBA"))
            if len(dimensions) != 1:
                raise Pause(f"GIF source PNGs for {name} do not share dimensions: {sorted(dimensions)}")
            usable.sort(key=lambda row: (float(row.get("midpoint_jd", 0.0)), row.get("source", "")))
            output = gif_product_path(state, usable[0])
            output.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(output, format="GIF", save_all=True, append_images=images[1:],
                           duration=GIF_DURATION_MS, loop=0, disposal=2)
            with Image.open(output) as gif:
                frame_count = sum(1 for _ in ImageSequence.Iterator(gif))
                gif_size = gif.size
                durations = []
                for frame in ImageSequence.Iterator(gif):
                    durations.append(frame.info.get("duration"))
            if frame_count != len(usable) or gif_size != next(iter(dimensions)) or any(d != GIF_DURATION_MS for d in durations):
                raise Pause(f"GIF audit failed for {output}: frames={frame_count}, size={gif_size}, durations={durations}")
            receipt(state, [output])
            for row in rows:
                row.update(gif="success", gif_path=str(output))
        finally:
            for image in images:
                image.close()
        checkpoint(state)
        print(f"GIFs {index}/{len(by_object)}: {name}: success", flush=True)


def audit(state):
    verify_sources(state)
    verify_receipts(state)
    results, helpers = cutout_helpers()
    for row in science_frames(state):
        if row["reduction"] != "success" or row["organization"] != "success":
            raise Pause(f"Incomplete processing: {row['source']}")
        original = fits.getdata(row["reduced_path"])
        organized = fits.getdata(row["organized_path"])
        if not np.array_equal(original, organized, equal_nan=True):
            raise Pause(f"Organized science pixel values changed: {row['source']}")
        if row["cutout"] not in {"success", "warning"}:
            if row["solve"] == "failed" and row["cutout"] == "skipped":
                continue
            raise Pause(f"Unaccounted cutout: {row['source']}")
        with fits.open(row["cutout_fits"], memmap=False) as hdus:
            hdus.verify("exception")
            header, data = hdus[0].header, hdus[0].data
            expected, wcs, _, info = helpers._build_cutout(
                row["organized_path"], header["CUTRA"], header["CUTDEC"],
                CUTOUT_SIZE_ARCSEC, size_px=CUTOUT_PIXELS)
            transform = dict(results._DISCRETE_ARRAY_TRANSFORMS)[header["CUTXFM"]]
            if not np.array_equal(data, transform(expected), equal_nan=True) or header["CUTRSMP"]:
                raise Pause(f"Cutout does not preserve source pixels: {row['cutout_fits']}")
            x, y = WCS(header).world_to_pixel_values(header["CUTRA"], header["CUTDEC"])
            if not info["inside"] or not info["overlap"] or abs(x - (data.shape[1] - 1) / 2) > 1 or abs(y - (data.shape[0] - 1) / 2) > 1:
                raise Pause(f"Cutout is not centered at the predicted object: {row['cutout_fits']}")
    gif_count = 0
    for name in sorted({r["object"] for r in science_frames(state)}):
        rows = [r for r in science_frames(state) if r["object"] == name]
        if len(rows) == 1 and rows[0].get("gif") == "skipped":
            continue
        if not all(r.get("gif") == "success" for r in rows):
            raise Pause(f"GIF stage incomplete for {name}")
        path = Path(rows[0]["gif_path"])
        if not path.is_file():
            raise Pause(f"Missing GIF for {name}: {path}")
        from PIL import Image, ImageSequence
        png_sizes = {Image.open(r["cutout_png"]).size for r in rows if r.get("cutout_png")}
        with Image.open(path) as gif:
            durations = [frame.info.get("duration") for frame in ImageSequence.Iterator(gif)]
            if gif.size not in png_sizes or any(d != GIF_DURATION_MS for d in durations):
                raise Pause(f"GIF dimensions/timing audit failed: {path}")
        gif_count += 1
    return {"source_hashes_unchanged": True, "pixel_arrays_verified": True,
            "science_count": len(science_frames(state)), "gif_count": gif_count,
            "gif_duration_ms": GIF_DURATION_MS, "objects": object_summary(state)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reduced", nargs="?")
    parser.add_argument("--master")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-map", help="JSON object mapping OBJNAME values to resolved Horizons identifiers")
    parser.add_argument("--probe-file", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.probe_file:
        print(json.dumps(probe_file(args.probe_file)))
        return 0
    if not args.reduced:
        raise Pause("A reduced-data path is required")
    reduced, master, date = discover_master(args.reduced, args.master)
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else reduced.parent / "superflat_processed"
    if root == reduced or root == master or root.is_relative_to(reduced) or root.is_relative_to(master) or reduced.is_relative_to(root) or master.is_relative_to(root):
        raise Pause("Output must be separate from raw and existing reduced trees")
    settings = {"flat_mode": "superflat", "bad_amp": "auto", "cutout_arcsec": CUTOUT_SIZE_ARCSEC,
                "cutout_pixels": CUTOUT_PIXELS, "site": "705",
                "gif_duration_ms": GIF_DURATION_MS, "naming_version": 2,
                "target_map": json.loads(Path(args.target_map).read_text()) if args.target_map else {}}
    if not isinstance(settings["target_map"], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in settings["target_map"].items()):
        raise Pause("Target map must contain string names and string Horizons identifiers")
    if args.preflight_only:
        report = preflight(reduced, master, date)
        print(json.dumps({k: v for k, v in report.items() if k not in {"frames", "original_reduced"}}, indent=2))
        return 0
    state = None
    lock = None
    try:
        if root.exists() and not args.resume:
            raise Pause(f"Output already exists; inspect it and use --resume: {root}")
        root.mkdir(parents=True, exist_ok=True)
        lock = root / ".pipeline.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            lock = None
            raise Pause("Pipeline lock exists; check the recorded PID before recovering an interrupted run")
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
        for name in ("logs", "astrometry", "ephemerides"):
            (root / name).mkdir(exist_ok=True)
        if args.resume:
            path = root / "pipeline_state.json"
            if not path.is_file():
                raise Pause("Cannot resume without pipeline_state.json; inspect the incomplete output directory")
            state = json.loads(path.read_text())
            if args.target_map is None:
                settings["target_map"] = state["settings"]["target_map"]
            if (state["reduced"], state["master"], state["output"]) != (str(reduced), str(master), str(root)):
                raise Pause("Resume paths/settings differ from the saved run")
            old_settings = state["settings"]
            old_settings.setdefault("gif_duration_ms", GIF_DURATION_MS)
            old_settings.setdefault("cutout_pixels", CUTOUT_PIXELS)
            old_settings.setdefault("naming_version", 2)
            for row in state["frames"]:
                row.setdefault("gif", "pending" if row.get("image_type") == "Object" and not row.get("excluded") else ("excluded" if row.get("excluded") else "not_applicable"))
            if {k: v for k, v in old_settings.items() if k != "target_map"} != {k: v for k, v in settings.items() if k != "target_map"}:
                raise Pause("Resume settings differ from the saved run")
            changed_targets = {key for key in set(old_settings["target_map"]) | set(settings["target_map"])
                               if old_settings["target_map"].get(key) != settings["target_map"].get(key)}
            if any(row.get("object") in changed_targets and row.get("ephemeris_id") for row in state["frames"]):
                raise Pause("Cannot change an ephemeris identifier after it has produced results")
            state["settings"] = settings
            if state.get("preflight_complete"):
                verify_sources(state)
                if state.get("executions") and state["executions"][0]["acronym_sha256"] != digest(REPO / "acronym.py"):
                    raise Pause("Acronym reduction code changed since this run; inspect compatibility before resuming")
            elif snapshot(reduced) != state["original_reduced"]:
                raise Pause("The original reduced data changed during interrupted preflight")
            verify_receipts(state)
            if state.get("preflight_complete") and digest(state["astrometry_config"]) != state["astrometry_config_sha256"]:
                raise Pause("Astrometry configuration changed since preflight")
        else:
            state = {"version": 1, "reduced": str(reduced), "master": str(master), "output": str(root),
                     "date": date, "settings": settings, "artifacts": {}, "status": "running", "frames": [],
                     "original_reduced": snapshot(reduced)}
        if not state.get("preflight_complete"):
            def progress(rows):
                state["frames"] = rows
                checkpoint(state)
            checkpoint(state)
            state.update(preflight(reduced, master, date, progress=progress, original_reduced=state["original_reduced"]))
            state["preflight_complete"] = True
        state.setdefault("executions", []).append({"utc": datetime.now(timezone.utc).isoformat(),
                                                  "runner_sha256": digest(__file__), "acronym_sha256": digest(REPO / "acronym.py"),
                                                  "python": sys.executable})
        state["status"] = "running"
        state.pop("error", None)
        checkpoint(state)
        reduce_and_organize(state)
        solve_frames(state)
        rename_products(state)
        cutout_frames(state)
        make_gifs(state)
        print("Auditing sources, manifests, WCS, and native-pixel products", flush=True)
        report = audit(state)
        atomic_json(root / "validation.json", report)
        state["status"] = "complete_with_warnings" if any(r["solve"] != "success" or r["cutout"] != "success" or r["gif"] not in {"success", "skipped"} or r.get("flat_mode") == "lamp-fallback" for r in science_frames(state)) else "complete"
        checkpoint(state)
        print(f"{state['status']}: {root / 'master_manifest.csv'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if state is not None:
            state["status"] = "paused"
            state["error"] = str(exc) or "Interrupted"
            checkpoint(state)
        print(f"PAUSED: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Pause as error:
        print(f"PAUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
