# coding: utf-8
"""Automatic reductions for ARCTIC imaging data.

The historical command remains valid::

    python acronym.py [directory]

Lamp flats remain the default.  ``--flat-mode superflat`` additionally builds
science-derived and hybrid lamp/science flats, with conservative lamp fallback
when too few independent boresights are available.
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, mad_std, sigma_clip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.spatial.distance import squareform


SATURATION_LEVEL = 65_000.0
AMPLIFIER_SUFFIXES = {
    "LL": "11",
    "LR": "21",
    "UL": "12",
    "UR": "22",
}
SUFFIX_AMPLIFIERS = {suffix: name for name, suffix in AMPLIFIER_SUFFIXES.items()}
BAD_AMP_CHOICES = ("auto", "none", *AMPLIFIER_SUFFIXES)


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    objtype: str
    filt: str
    exposure: float
    objname: str
    readamps: str
    binning: tuple[int | None, int | None]
    ra: str | float | None
    dec: str | float | None
    header: fits.Header
    bad_amplifiers: tuple[str, ...] = ()

    @property
    def config_key(self) -> tuple[object, ...]:
        section_keys = tuple(
            f"{prefix}{suffix}"
            for prefix in ("DSEC", "BSEC")
            for suffix in ("11", "21", "12", "22")
        )
        sections = tuple(self.header.get(key, "") for key in section_keys)
        return (
            self.readamps,
            *self.binning,
            self.header.get("NAXIS1"),
            self.header.get("NAXIS2"),
            *sections,
        )


@dataclass(frozen=True)
class BoresightAssignment:
    group_id: int
    separation_arcsec: float


@dataclass(frozen=True)
class CandidateMetric:
    record: FrameRecord
    usable_fraction: float
    sky_median: float
    sky_noise: float
    sky_snr: float
    valid: bool


def normalize_filter(raw_filter: object) -> str:
    """Normalize ARCTIC filter labels while retaining multi-letter filters."""
    value = "" if raw_filter is None else str(raw_filter)
    value = value.replace("SDSS", "").strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value


def clean_filtname(value: object) -> str:
    """Return the stable filename token used for a filter."""
    return normalize_filter(value).replace(" ", "").replace("#", "")


def _parse_section(section: str) -> tuple[slice, slice]:
    match = re.fullmatch(
        r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*,\s*(-?\d+)\s*:\s*(-?\d+)\s*\]",
        section,
    )
    if match is None:
        raise ValueError(f"Unsupported FITS section: {section!r}")
    x1, x2, y1, y2 = (int(value) for value in match.groups())
    xstep = 1 if x2 >= x1 else -1
    ystep = 1 if y2 >= y1 else -1
    xslice = slice(x1 - 1, x2 if xstep == 1 else x2 - 2, xstep)
    yslice = slice(y1 - 1, y2 if ystep == 1 else y2 - 2, ystep)
    return yslice, xslice


def _extract_section(data: np.ndarray, section: str) -> np.ndarray:
    yslice, xslice = _parse_section(section)
    return np.asarray(data[yslice, xslice], dtype=np.float32)


def _overscan_correct(
    science: np.ndarray, overscan: np.ndarray, polynomial_order: int = 3
) -> np.ndarray:
    if science.shape[0] != overscan.shape[0]:
        raise ValueError(
            "Science and overscan sections have different row counts: "
            f"{science.shape[0]} != {overscan.shape[0]}"
        )
    row_bias = np.nanmedian(overscan, axis=1)
    clipped = sigma_clip(row_bias, sigma=4.0, maxiters=3, masked=True)
    good = ~np.ma.getmaskarray(clipped) & np.isfinite(row_bias)
    rows = np.arange(science.shape[0], dtype=float)
    if np.count_nonzero(good) >= 2:
        degree = min(polynomial_order, np.count_nonzero(good) - 1)
        model = np.polyval(np.polyfit(rows[good], row_bias[good], degree), rows)
    else:
        model = np.full(science.shape[0], np.nanmedian(row_bias))
    return np.asarray(science, dtype=np.float32) - model[:, None].astype(np.float32)


def _validate_bad_amp(bad_amp: str) -> str:
    value = str(bad_amp).strip()
    if value not in BAD_AMP_CHOICES:
        choices = ", ".join(BAD_AMP_CHOICES)
        raise ValueError(f"Unsupported bad_amp value {bad_amp!r}; choose from {choices}")
    return value


def _quad_bad_amplifiers(
    raw: np.ndarray, header: fits.Header, bad_amp: str
) -> tuple[str, ...]:
    mode = _validate_bad_amp(bad_amp)
    if str(header.get("READAMPS", "") or "") != "Quad" or mode == "none":
        return ()
    if mode != "auto":
        return (mode,)

    detected: list[str] = []
    for suffix in ("11", "21", "12", "22"):
        bias_key = f"BSEC{suffix}"
        if bias_key not in header:
            raise KeyError(f"Quad frame lacks {bias_key}")
        overscan = _extract_section(raw, header[bias_key])
        if float(np.nanmedian(overscan)) >= SATURATION_LEVEL:
            detected.append(SUFFIX_AMPLIFIERS[suffix])
    return tuple(detected)


def _bad_amplifier_value(records: Sequence[FrameRecord]) -> str | None:
    detected = {
        amplifier
        for record in records
        for amplifier in record.bad_amplifiers
    }
    ordered = [name for name in AMPLIFIER_SUFFIXES if name in detected]
    return " ".join(ordered) or None


def trim_image(
    filename: str | Path,
    overscan_poly_order: int = 3,
    bad_amp: str = "auto",
) -> tuple[np.ndarray, fits.Header]:
    """Overscan-correct and mosaic the active detector area of an ARCTIC frame."""
    # ARCTIC raw frames are unsigned integers represented with FITS BZERO/BSCALE;
    # Astropy must materialize those scaled values rather than memory-map them.
    with fits.open(filename, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()

    mode = _validate_bad_amp(bad_amp)
    readamps = str(header.get("READAMPS", "") or "")
    if readamps == "Quad":
        bad_amplifiers = _quad_bad_amplifiers(raw, header, mode)
        amplifiers: dict[str, np.ndarray] = {}
        for suffix in ("11", "21", "12", "22"):
            data_key = f"DSEC{suffix}"
            bias_key = f"BSEC{suffix}"
            if data_key not in header or bias_key not in header:
                raise KeyError(f"Quad frame lacks {data_key} or {bias_key}")
            science = _extract_section(raw, header[data_key])
            overscan = _extract_section(raw, header[bias_key])
            amplifier_name = SUFFIX_AMPLIFIERS[suffix]
            if amplifier_name in bad_amplifiers:
                amplifiers[suffix] = np.full(science.shape, np.nan, dtype=np.float32)
            else:
                amplifiers[suffix] = _overscan_correct(
                    science, overscan, polynomial_order=overscan_poly_order
                )
        upper = np.concatenate((amplifiers["11"], amplifiers["21"]), axis=1)
        lower = np.concatenate((amplifiers["12"], amplifiers["22"]), axis=1)
        image = np.concatenate((upper, lower), axis=0)
    elif readamps in AMPLIFIER_SUFFIXES:
        bad_amplifiers = ()
        suffix = AMPLIFIER_SUFFIXES[readamps]
        data_key = f"DSEC{suffix}"
        bias_key = f"BSEC{suffix}"
        if data_key not in header or bias_key not in header:
            raise KeyError(f"{readamps} frame lacks {data_key} or {bias_key}")
        science = _extract_section(raw, header[data_key])
        overscan = _extract_section(raw, header[bias_key])
        image = _overscan_correct(
            science, overscan, polynomial_order=overscan_poly_order
        )
    else:
        raise ValueError(f"Unsupported READAMPS value {readamps!r} in {filename}")

    header["OVSCORR"] = (True, "Healthy amplifier overscan correction applied")
    if bad_amplifiers:
        names = " ".join(bad_amplifiers)
        header["BADAMPS"] = (names, "Amplifiers replaced with NaN before calibration")
        for name in bad_amplifiers:
            suffix = AMPLIFIER_SUFFIXES[name]
            header.add_history(
                f"Acronym set amplifier {name} ({suffix}) data to NaN before calibration"
            )
    return np.asarray(image, dtype=np.float32), header


def catalog_frames(
    directory: str | Path, bad_amp: str = "auto"
) -> tuple[list[FrameRecord], list[tuple[Path, str]]]:
    mode = _validate_bad_amp(bad_amp)
    records: list[FrameRecord] = []
    failures: list[tuple[Path, str]] = []
    for path in sorted(Path(directory).glob("*.fits")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with fits.open(path, memmap=False) as hdul:
                    header = hdul[0].header.copy()
                    if hdul[0].data is None:
                        raise ValueError("primary HDU has no image data")
                    raw = np.asarray(hdul[0].data)
            if raw.ndim != 2 or raw.size == 0:
                raise ValueError(f"primary image has invalid shape {raw.shape}")
            bad_amplifiers = _quad_bad_amplifiers(raw, header, mode)
            exposure = float(header.get("EXPTIME", 0.0) or 0.0)
            records.append(
                FrameRecord(
                    path=path,
                    objtype=str(header.get("IMAGETYP", "") or ""),
                    filt=normalize_filter(header.get("FILTER", "")),
                    exposure=exposure,
                    objname=str(header.get("OBJNAME", "") or ""),
                    readamps=str(header.get("READAMPS", "") or ""),
                    binning=(header.get("CCDBIN1"), header.get("CCDBIN2")),
                    ra=header.get("RA"),
                    dec=header.get("DEC"),
                    header=header,
                    bad_amplifiers=bad_amplifiers,
                )
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures.append((path, str(exc)))
    return records, failures


def _validate_detector_config(records: Sequence[FrameRecord]) -> None:
    configs = {record.config_key for record in records}
    if len(configs) > 1:
        readout_modes = ", ".join(sorted({record.readamps for record in records}))
        raise ValueError(
            "Multiple detector configurations were found "
            f"({readout_modes}); separate those configurations before reduction."
        )


def _robust_stats(data: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    finite_mask = ~np.isfinite(data)
    if mask is not None:
        finite_mask |= mask
    if np.all(finite_mask):
        return float("nan"), float("nan")
    _, median, stddev = sigma_clipped_stats(
        data, mask=finite_mask, sigma=3.0, maxiters=3, stdfunc=mad_std
    )
    return float(median), float(stddev)


def _normalize(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    median, _ = _robust_stats(data, mask)
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Cannot normalize an image with a nonpositive robust median")
    result = np.asarray(data, dtype=np.float32) / np.float32(median)
    if mask is not None:
        result = result.copy()
        result[mask] = np.nan
    return result


def sigma_clipped_median(
    images: Sequence[np.ndarray], sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    if not images:
        raise ValueError("At least one image is required for combination")
    stack = np.asarray(images, dtype=np.float32)
    masked = np.ma.masked_invalid(stack)
    clipped = sigma_clip(
        masked,
        sigma=sigma,
        maxiters=3,
        axis=0,
        cenfunc="median",
        stdfunc=mad_std,
        masked=True,
        copy=False,
    )
    result = np.ma.median(clipped, axis=0).filled(np.nan).astype(np.float32)
    coverage = np.sum(~np.ma.getmaskarray(clipped), axis=0).astype(np.int16)
    return result, coverage


def _calibration_header(caltype: str, **metadata: object) -> fits.Header:
    header = fits.Header()
    header["CALTYPE"] = (caltype, "Acronym calibration product")
    for key, value in metadata.items():
        if value is not None:
            header[key.upper()] = value
    if metadata.get("BADAMPS"):
        header.add_history(
            f"Acronym inputs set amplifier(s) {metadata['BADAMPS']} to NaN before calibration"
        )
    header.add_history("Created by Acronym ARCTIC reduction pipeline")
    return header


def _write_fits(path: Path, data: np.ndarray, header: fits.Header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=header, overwrite=True)


def _subtract_bias(data: np.ndarray, bias: np.ndarray | float) -> np.ndarray:
    return np.asarray(data, dtype=np.float32) - bias


def build_master_bias(
    records: Sequence[FrameRecord],
    calibration_directory: Path,
    bad_amp: str = "auto",
) -> np.ndarray | float:
    bias_records = [record for record in records if record.objtype == "Bias"]
    if not bias_records:
        print("   > No biases found. Continuing reductions...")
        return 0.0
    images = [trim_image(record.path, bad_amp=bad_amp)[0] for record in bias_records]
    master, _ = sigma_clipped_median(images, sigma=5.0)
    path = calibration_directory / "master_bias.fits"
    header = _calibration_header(
        "MASTER BIAS",
        NCOMB=len(images),
        OVSCORR=True,
        BADAMPS=_bad_amplifier_value(bias_records),
    )
    _write_fits(path, master, header)
    print(f"   > Created master bias: {path}")
    return master


def build_master_darks(
    records: Sequence[FrameRecord],
    bias: np.ndarray | float,
    calibration_directory: Path,
    bad_amp: str = "auto",
) -> dict[float, np.ndarray]:
    grouped: dict[float, list[FrameRecord]] = defaultdict(list)
    for record in records:
        if record.objtype == "Dark":
            grouped[record.exposure].append(record)
    masters: dict[float, np.ndarray] = {}
    for exposure in sorted(grouped):
        images = [
            _subtract_bias(trim_image(record.path, bad_amp=bad_amp)[0], bias)
            for record in grouped[exposure]
        ]
        master, _ = sigma_clipped_median(images, sigma=5.0)
        masters[exposure] = master
        path = calibration_directory / f"master_dark_{exposure}.fits"
        header = _calibration_header(
            "MASTER DARK",
            EXPTIME=exposure,
            NCOMB=len(images),
            BIASCOR=True,
            OVSCORR=True,
            BADAMPS=_bad_amplifier_value(grouped[exposure]),
        )
        _write_fits(path, master, header)
        print(f"   > Created master {exposure} second dark: {path}")
    if not masters:
        print("   > No darks found. Continuing reductions...")
    return masters


def select_dark(
    exposure: float, darks: dict[float, np.ndarray]
) -> tuple[np.ndarray | None, float | None]:
    for dark_exposure, dark in darks.items():
        if np.isclose(exposure, dark_exposure, rtol=0.0, atol=1.0e-6):
            return dark, dark_exposure
    if not darks:
        return None, None
    source_exposure = max(darks)
    if source_exposure <= 0:
        return None, None
    return darks[source_exposure] * np.float32(exposure / source_exposure), source_exposure


def calibrate_additive_signals(
    record: FrameRecord,
    bias: np.ndarray | float,
    darks: dict[float, np.ndarray],
    bad_amp: str = "auto",
) -> tuple[np.ndarray, fits.Header, float | None, np.ndarray]:
    trimmed, header = trim_image(record.path, bad_amp=bad_amp)
    saturated = trimmed >= SATURATION_LEVEL
    calibrated = _subtract_bias(trimmed, bias)
    dark, source_exposure = select_dark(record.exposure, darks)
    if dark is not None:
        calibrated = calibrated - dark
    return calibrated.astype(np.float32), header, source_exposure, saturated


def build_lamp_flats(
    records: Sequence[FrameRecord],
    bias: np.ndarray | float,
    darks: dict[float, np.ndarray],
    bad_amp: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        if record.objtype == "Flat" and record.filt:
            grouped[record.filt].append(record)
    masters: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for filt_name in sorted(grouped):
        normalized: list[np.ndarray] = []
        for record in grouped[filt_name]:
            calibrated, _, _, saturated = calibrate_additive_signals(
                record, bias, darks, bad_amp=bad_amp
            )
            invalid = saturated | ~np.isfinite(calibrated) | (calibrated <= 0)
            normalized.append(_normalize(calibrated, invalid))
        master, _ = sigma_clipped_median(normalized, sigma=5.0)
        master = _normalize(master)
        masters[filt_name] = master
        counts[filt_name] = len(normalized)
    return masters, counts


def _as_coordinate(record: FrameRecord) -> SkyCoord | None:
    if record.ra in (None, "") or record.dec in (None, ""):
        return None
    try:
        return SkyCoord(record.ra, record.dec, unit=(u.hourangle, u.deg), frame="icrs")
    except (ValueError, TypeError, u.UnitsError):
        return None


def cluster_boresights(
    records: Sequence[FrameRecord], threshold_arcsec: float
) -> tuple[dict[Path, BoresightAssignment], int]:
    ordered = sorted(records, key=lambda record: record.path.name)
    valid = [(record, _as_coordinate(record)) for record in ordered]
    valid = [(record, coord) for record, coord in valid if coord is not None]
    if not valid:
        return {}, 0
    if len(valid) == 1:
        return {valid[0][0].path: BoresightAssignment(1, 0.0)}, 1

    coordinates = SkyCoord([coord for _, coord in valid])
    separations = coordinates[:, None].separation(coordinates[None, :]).arcsec
    np.fill_diagonal(separations, 0.0)
    hierarchy = linkage(squareform(separations, checks=False), method="complete")
    raw_labels = fcluster(hierarchy, t=threshold_arcsec, criterion="distance")

    raw_groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(raw_labels):
        raw_groups[int(label)].append(index)
    sorted_groups = sorted(
        raw_groups.values(), key=lambda indices: valid[min(indices)][0].path.name
    )

    assignments: dict[Path, BoresightAssignment] = {}
    for group_id, indices in enumerate(sorted_groups, start=1):
        xyz = np.asarray([coordinates[index].cartesian.xyz.value for index in indices])
        mean_xyz = np.mean(xyz, axis=0)
        mean_xyz /= np.linalg.norm(mean_xyz)
        center = SkyCoord(
            x=mean_xyz[0],
            y=mean_xyz[1],
            z=mean_xyz[2],
            representation_type="cartesian",
            frame="icrs",
        )
        for index in indices:
            record = valid[index][0]
            separation = float(coordinates[index].separation(center).arcsec)
            assignments[record.path] = BoresightAssignment(group_id, separation)
    return assignments, len(sorted_groups)


def make_source_mask(
    image: np.ndarray,
    sigma_threshold: float = 3.0,
    dilation: int = 15,
) -> tuple[np.ndarray, float, float]:
    data = np.asarray(image, dtype=np.float32)
    base_mask = ~np.isfinite(data) | (data <= 0) | (data >= SATURATION_LEVEL)
    ny, nx = data.shape
    box_size = (max(4, min(64, ny // 4)), max(4, min(64, nx // 4)))
    background: np.ndarray
    background_rms: np.ndarray
    try:
        estimator = Background2D(
            data,
            box_size=box_size,
            filter_size=(3, 3),
            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
            bkg_estimator=MedianBackground(),
            mask=base_mask,
            exclude_percentile=90.0,
        )
        background = np.asarray(estimator.background, dtype=np.float32)
        background_rms = np.asarray(estimator.background_rms, dtype=np.float32)
    except (ValueError, TypeError):
        median, noise = _robust_stats(data, base_mask)
        background = np.full(data.shape, median, dtype=np.float32)
        background_rms = np.full(data.shape, noise, dtype=np.float32)

    residual = data - background
    detection_image = gaussian_filter(np.where(base_mask, 0.0, residual), sigma=2.0)
    threshold = sigma_threshold * background_rms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segmentation = detect_sources(
            detection_image,
            threshold,
            npixels=5,
            connectivity=8,
            mask=base_mask,
        )
    source_mask = np.zeros(data.shape, dtype=bool)
    if segmentation is not None:
        source_mask = segmentation.data > 0
        if dilation > 0:
            source_mask = binary_dilation(source_mask, iterations=dilation)
    final_mask = base_mask | source_mask
    sky_median, sky_noise = _robust_stats(data, final_mask)
    return final_mask, sky_median, sky_noise


def measure_candidate(
    record: FrameRecord,
    bias: np.ndarray | float,
    darks: dict[float, np.ndarray],
    lamp_flat: np.ndarray,
    mask_sigma: float,
    mask_dilation: int,
    bad_amp: str = "auto",
) -> CandidateMetric:
    calibrated, _, _, saturated = calibrate_additive_signals(
        record, bias, darks, bad_amp=bad_amp
    )
    lamp_corrected = np.full(calibrated.shape, np.nan, dtype=np.float32)
    valid_flat = np.isfinite(lamp_flat) & (lamp_flat > 0)
    lamp_corrected[valid_flat] = calibrated[valid_flat] / lamp_flat[valid_flat]
    mask, sky_median, sky_noise = make_source_mask(
        lamp_corrected, sigma_threshold=mask_sigma, dilation=mask_dilation
    )
    mask |= saturated
    usable_fraction = float(1.0 - np.mean(mask))
    if np.isfinite(sky_median) and sky_median > 0 and sky_noise == 0:
        sky_snr = float("inf")
    elif np.isfinite(sky_median) and np.isfinite(sky_noise) and sky_noise > 0:
        sky_snr = float(sky_median / sky_noise)
    else:
        sky_snr = float("nan")
    valid = usable_fraction >= 0.5 and not np.isnan(sky_snr) and sky_snr > 0
    return CandidateMetric(
        record=record,
        usable_fraction=usable_fraction,
        sky_median=sky_median,
        sky_noise=sky_noise,
        sky_snr=sky_snr,
        valid=valid,
    )


def choose_representatives(
    candidates: Sequence[CandidateMetric],
    assignments: dict[Path, BoresightAssignment],
) -> dict[int, CandidateMetric]:
    grouped: dict[int, list[CandidateMetric]] = defaultdict(list)
    for candidate in candidates:
        assignment = assignments.get(candidate.record.path)
        if assignment is not None and candidate.valid:
            grouped[assignment.group_id].append(candidate)
    selected: dict[int, CandidateMetric] = {}
    for group_id, group in grouped.items():
        selected[group_id] = sorted(
            group,
            key=lambda item: (
                -item.usable_fraction,
                -item.sky_snr,
                item.record.path.name,
            ),
        )[0]
    return selected


def smooth_masked_image(image: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(image)
    values = gaussian_filter(np.where(valid, image, 0.0), sigma=sigma)
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma)
    result = np.ones(image.shape, dtype=np.float32)
    supported = weights > 1.0e-6
    result[supported] = values[supported] / weights[supported]
    return _normalize(result)


def combine_superflat_inputs(
    raw_science: Sequence[np.ndarray],
    residual_science: Sequence[np.ndarray],
    lamp_flat: np.ndarray,
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_science_flat, raw_coverage = sigma_clipped_median(raw_science, sigma=3.0)
    full_science_flat = _normalize(full_science_flat)
    residual, residual_coverage = sigma_clipped_median(residual_science, sigma=3.0)
    residual = _normalize(residual)
    illumination = smooth_masked_image(residual, sigma=smoothing_sigma)
    hybrid = _normalize(lamp_flat * illumination)
    coverage = np.minimum(raw_coverage, residual_coverage).astype(np.int16)
    return full_science_flat, illumination, hybrid, coverage


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_superflats(
    science_records: Sequence[FrameRecord],
    lamp_flats: dict[str, np.ndarray],
    lamp_counts: dict[str, int],
    bias: np.ndarray | float,
    darks: dict[float, np.ndarray],
    calibration_directory: Path,
    boresight_threshold: float,
    minimum_boresights: int,
    mask_sigma: float,
    mask_dilation: int,
    smoothing_sigma: float,
    bad_amp: str = "auto",
    bad_amplifiers: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[dict[str, object]]]:
    applied = dict(lamp_flats)
    applied_modes = {filt_name: "lamp-fallback" for filt_name in lamp_flats}
    manifest_rows: list[dict[str, object]] = []

    for filt_name, lamp_flat in sorted(lamp_flats.items()):
        token = clean_filtname(filt_name)
        lamp_header = _calibration_header(
            "MASTER LAMP FLAT",
            FILTER=filt_name,
            FLATMODE="lamp",
            NCOMB=lamp_counts[filt_name],
            BIASCOR=True,
            DARKCOR=True,
            OVSCORR=True,
            BADAMPS=bad_amplifiers,
        )
        _write_fits(calibration_directory / f"master_lamp_flat_{token}.fits", lamp_flat, lamp_header)

        filter_records = [record for record in science_records if record.filt == filt_name]
        assignments, raw_group_count = cluster_boresights(
            filter_records, threshold_arcsec=boresight_threshold
        )
        candidates = [
            measure_candidate(
                record,
                bias,
                darks,
                lamp_flat,
                mask_sigma=mask_sigma,
                mask_dilation=mask_dilation,
                bad_amp=bad_amp,
            )
            for record in filter_records
        ]
        representatives = choose_representatives(candidates, assignments)
        selected_paths = {candidate.record.path for candidate in representatives.values()}
        eligible = len(representatives) >= minimum_boresights
        for candidate in candidates:
            assignment = assignments.get(candidate.record.path)
            is_representative = candidate.record.path in selected_paths
            if not candidate.valid:
                status = "rejected_quality"
                reason = "less than 50% usable area or invalid sky statistics"
            elif not eligible:
                status = "fallback_insufficient_boresights"
                reason = f"{len(representatives)} usable boresights < {minimum_boresights}"
            elif is_representative:
                status = "selected"
                reason = ""
            else:
                status = "excluded_duplicate_boresight"
                reason = "higher-quality representative selected for this boresight"
            manifest_rows.append(
                {
                    "filename": candidate.record.path.name,
                    "filter": filt_name,
                    "object": candidate.record.objname,
                    "ra": candidate.record.ra or "",
                    "dec": candidate.record.dec or "",
                    "boresight_id": assignment.group_id if assignment else "",
                    "separation_arcsec": (
                        f"{assignment.separation_arcsec:.6f}" if assignment else ""
                    ),
                    "representative": is_representative,
                    "selected_for_superflat": bool(is_representative and eligible),
                    "usable_fraction": f"{candidate.usable_fraction:.8f}",
                    "sky_median": f"{candidate.sky_median:.8g}",
                    "sky_noise": f"{candidate.sky_noise:.8g}",
                    "sky_snr": f"{candidate.sky_snr:.8g}",
                    "status": status,
                    "reason": reason,
                }
            )

        if not eligible:
            print(
                f"   > {filt_name}: {len(representatives)} usable boresights "
                f"({raw_group_count} coordinate groups); using lamp flat"
            )
            continue

        raw_stack: list[np.ndarray] = []
        residual_stack: list[np.ndarray] = []
        for group_id in sorted(representatives):
            record = representatives[group_id].record
            calibrated, _, _, saturated = calibrate_additive_signals(
                record, bias, darks, bad_amp=bad_amp
            )
            residual = np.full(calibrated.shape, np.nan, dtype=np.float32)
            valid_flat = np.isfinite(lamp_flat) & (lamp_flat > 0)
            residual[valid_flat] = calibrated[valid_flat] / lamp_flat[valid_flat]
            mask, _, _ = make_source_mask(
                residual, sigma_threshold=mask_sigma, dilation=mask_dilation
            )
            mask |= saturated | ~np.isfinite(calibrated) | (calibrated <= 0)
            raw_stack.append(_normalize(calibrated, mask))
            residual_stack.append(_normalize(residual, mask))

        full_science, illumination, hybrid, coverage = combine_superflat_inputs(
            raw_stack,
            residual_stack,
            lamp_flat,
            smoothing_sigma=smoothing_sigma,
        )
        metadata = {
            "FILTER": filt_name,
            "FLATMODE": "hybrid",
            "NBORE": len(representatives),
            "BORESEP": boresight_threshold,
            "MASKSIG": mask_sigma,
            "MASKDIL": mask_dilation,
            "SMTHSIG": smoothing_sigma,
            "BIASCOR": True,
            "DARKCOR": True,
            "OVSCORR": True,
            "BADAMPS": bad_amplifiers,
        }
        _write_fits(
            calibration_directory / f"master_science_flat_{token}.fits",
            full_science,
            _calibration_header("MASTER SCIENCE FLAT", **metadata),
        )
        _write_fits(
            calibration_directory / f"illumination_correction_{token}.fits",
            illumination,
            _calibration_header("ILLUMINATION CORRECTION", **metadata),
        )
        _write_fits(
            calibration_directory / f"master_superflat_{token}.fits",
            hybrid,
            _calibration_header("MASTER SUPERFLAT", **metadata),
        )
        coverage_header = _calibration_header(
            "SUPERFLAT COVERAGE",
            FILTER=filt_name,
            NBORE=len(representatives),
            BORESEP=boresight_threshold,
            BADAMPS=bad_amplifiers,
        )
        _write_fits(
            calibration_directory / f"superflat_coverage_{token}.fits",
            coverage,
            coverage_header,
        )
        applied[filt_name] = hybrid
        applied_modes[filt_name] = "hybrid"
        print(f"   > {filt_name}: created hybrid superflat from {len(representatives)} boresights")

    manifest_fields = (
        "filename",
        "filter",
        "object",
        "ra",
        "dec",
        "boresight_id",
        "separation_arcsec",
        "representative",
        "selected_for_superflat",
        "usable_fraction",
        "sky_median",
        "sky_noise",
        "sky_snr",
        "status",
        "reason",
    )
    _write_csv(calibration_directory / "superflat_manifest.csv", manifest_rows, manifest_fields)
    return applied, applied_modes, manifest_rows


def write_applied_flats(
    flats_by_filter: dict[str, np.ndarray],
    modes_by_filter: dict[str, str],
    lamp_counts: dict[str, int],
    calibration_directory: Path,
    superflat_manifest: Sequence[dict[str, object]] = (),
    boresight_threshold: float | None = None,
    mask_sigma: float | None = None,
    mask_dilation: int | None = None,
    smoothing_sigma: float | None = None,
    bad_amplifiers: str | None = None,
) -> None:
    boresight_counts: dict[str, int] = defaultdict(int)
    for row in superflat_manifest:
        if row.get("representative") is True:
            boresight_counts[str(row["filter"])] += 1
    for filt_name, flat in sorted(flats_by_filter.items()):
        token = clean_filtname(filt_name)
        mode = modes_by_filter[filt_name]
        header = _calibration_header(
            "MASTER FLAT",
            FILTER=filt_name,
            FLATMODE=mode,
            NCOMB=lamp_counts.get(filt_name, 0),
            BIASCOR=True,
            DARKCOR=True,
            OVSCORR=True,
            BADAMPS=bad_amplifiers,
        )
        if mode == "hybrid":
            header["SOURCE"] = f"master_superflat_{token}.fits"
        elif mode == "lamp-fallback":
            header["SOURCE"] = f"master_lamp_flat_{token}.fits"
        else:
            header["SOURCE"] = "combined lamp flats"
        if superflat_manifest:
            header["NBORE"] = boresight_counts.get(filt_name, 0)
            header["BORESEP"] = boresight_threshold
            header["MASKSIG"] = mask_sigma
            header["MASKDIL"] = mask_dilation
            header["SMTHSIG"] = smoothing_sigma
        _write_fits(calibration_directory / f"master_flat_{token}.fits", flat, header)


def tile_background_metrics(
    image: np.ndarray, mask: np.ndarray, tile_count: int = 16
) -> tuple[float, float, int]:
    tile_medians: list[float] = []
    row_indices = np.array_split(np.arange(image.shape[0]), tile_count)
    column_indices = np.array_split(np.arange(image.shape[1]), tile_count)
    for rows in row_indices:
        for columns in column_indices:
            block = image[np.ix_(rows, columns)]
            block_mask = mask[np.ix_(rows, columns)]
            values = block[~block_mask & np.isfinite(block)]
            if values.size >= max(10, int(block.size * 0.1)):
                tile_medians.append(float(np.median(values)))
    if not tile_medians:
        return float("nan"), float("nan"), 0
    values = np.asarray(tile_medians)
    center = float(np.median(values))
    if center == 0 or not np.isfinite(center):
        return float("nan"), float("nan"), len(values)
    scatter = float(1.4826 * np.median(np.abs(values - center)) / abs(center))
    low, high = np.percentile(values, [5, 95])
    peak_to_peak = float((high - low) / abs(center))
    return scatter, peak_to_peak, len(values)


def reduce_science_frames(
    science_records: Sequence[FrameRecord],
    bias: np.ndarray | float,
    darks: dict[float, np.ndarray],
    applied_flats: dict[str, np.ndarray],
    applied_modes: dict[str, str],
    data_directory: Path,
    calibration_directory: Path,
    mask_sigma: float,
    mask_dilation: int,
    bad_amp: str = "auto",
    bad_amplifiers: str | None = None,
) -> list[dict[str, object]]:
    qa_rows: list[dict[str, object]] = []
    for record in science_records:
        calibrated, header, dark_exposure, saturated = calibrate_additive_signals(
            record, bias, darks, bad_amp=bad_amp
        )
        flat = applied_flats.get(record.filt)
        mode = applied_modes.get(record.filt, "unity")
        if flat is None:
            print(f"   > Warning! No {record.filt} flat found for {record.path}")
            reduced = calibrated
            flat_filename = "NONE"
        else:
            valid = np.isfinite(flat) & (flat > 0)
            reduced = np.full(calibrated.shape, np.nan, dtype=np.float32)
            reduced[valid] = calibrated[valid] / flat[valid]
            flat_filename = f"master_flat_{clean_filtname(record.filt)}.fits"

        header["BIASCOR"] = (True, "Master bias subtracted")
        header["DARKCOR"] = (dark_exposure is not None, "Master dark subtracted")
        if dark_exposure is not None:
            header["DARKEXP"] = (dark_exposure, "Exposure of source master dark")
        header["FLATMODE"] = (mode, "Flat-field mode applied")
        header["FLATFILE"] = (flat_filename, "Applied master flat")
        if bad_amplifiers is not None and header.get("BADAMPS") != bad_amplifiers:
            header["BADAMPS"] = (
                bad_amplifiers,
                "Amplifiers null in raw or calibration products",
            )
            header.add_history(
                f"Acronym calibration products set amplifier(s) {bad_amplifiers} to NaN"
            )
        header.add_history("Overscan, bias, dark, and flat corrections applied by Acronym")
        output_path = data_directory / f"red_{record.path.name}"
        _write_fits(output_path, reduced, header)

        mask, background, noise = make_source_mask(
            reduced, sigma_threshold=mask_sigma, dilation=mask_dilation
        )
        mask |= saturated
        scatter, peak_to_peak, tile_number = tile_background_metrics(reduced, mask)
        qa_rows.append(
            {
                "filename": record.path.name,
                "filter": record.filt,
                "object": record.objname,
                "flat_mode": mode,
                "mask_fraction": f"{np.mean(mask):.8f}",
                "background_median": f"{background:.8g}",
                "background_noise": f"{noise:.8g}",
                "tile_scatter_fraction": f"{scatter:.8g}",
                "tile_peak_to_peak_fraction": f"{peak_to_peak:.8g}",
                "usable_tiles": tile_number,
            }
        )
    fields = (
        "filename",
        "filter",
        "object",
        "flat_mode",
        "mask_fraction",
        "background_median",
        "background_noise",
        "tile_scatter_fraction",
        "tile_peak_to_peak_fraction",
        "usable_tiles",
    )
    _write_csv(calibration_directory / "calibration_qa.csv", qa_rows, fields)
    return qa_rows


def run_pipeline(
    directory: str | Path,
    flat_mode: str = "lamp",
    output_directory: str | Path | None = None,
    boresight_separation_arcsec: float = 15.0,
    min_superflat_boresights: int = 4,
    source_mask_sigma: float = 3.0,
    source_mask_dilation: int = 15,
    illumination_smoothing_sigma: float = 64.0,
    bad_amp: str = "auto",
) -> dict[str, object]:
    source_directory = Path(directory)
    output_root = (
        Path(output_directory) if output_directory is not None else source_directory / "reduced"
    )
    calibration_directory = output_root / "cals"
    data_directory = output_root / "data"

    records, failures = catalog_frames(source_directory, bad_amp=bad_amp)
    for path, reason in failures:
        print(f"   > Warning! Skipping unreadable FITS file {path}: {reason}")
    _validate_detector_config(records)
    bad_amplifiers = _bad_amplifier_value(records)
    if bad_amplifiers is not None:
        print(f"   > Nulling bad amplifier(s): {bad_amplifiers}")

    calibration_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)

    print("\n >>> Starting bias combine...")
    bias = build_master_bias(records, calibration_directory, bad_amp=bad_amp)

    print("\n >>> Starting darks...")
    darks = build_master_darks(
        records, bias, calibration_directory, bad_amp=bad_amp
    )

    print("\n >>> Starting lamp flats...")
    lamp_flats, lamp_counts = build_lamp_flats(
        records, bias, darks, bad_amp=bad_amp
    )
    print(f"   > Filters: {sorted(lamp_flats)}")

    science_records = [record for record in records if record.objtype == "Object"]
    if flat_mode == "superflat":
        print("\n >>> Starting science-derived superflats...")
        applied_flats, applied_modes, manifest = build_superflats(
            science_records,
            lamp_flats,
            lamp_counts,
            bias,
            darks,
            calibration_directory,
            boresight_threshold=boresight_separation_arcsec,
            minimum_boresights=min_superflat_boresights,
            mask_sigma=source_mask_sigma,
            mask_dilation=source_mask_dilation,
            smoothing_sigma=illumination_smoothing_sigma,
            bad_amp=bad_amp,
            bad_amplifiers=bad_amplifiers,
        )
    else:
        applied_flats = dict(lamp_flats)
        applied_modes = {filt_name: "lamp" for filt_name in lamp_flats}
        manifest = []

    write_applied_flats(
        applied_flats,
        applied_modes,
        lamp_counts,
        calibration_directory,
        superflat_manifest=manifest,
        boresight_threshold=(boresight_separation_arcsec if flat_mode == "superflat" else None),
        mask_sigma=(source_mask_sigma if flat_mode == "superflat" else None),
        mask_dilation=(source_mask_dilation if flat_mode == "superflat" else None),
        smoothing_sigma=(illumination_smoothing_sigma if flat_mode == "superflat" else None),
        bad_amplifiers=bad_amplifiers,
    )

    print(f"\n >>> {len(science_records)} science images found. Starting reductions...")
    qa_rows = reduce_science_frames(
        science_records,
        bias,
        darks,
        applied_flats,
        applied_modes,
        data_directory,
        calibration_directory,
        mask_sigma=source_mask_sigma,
        mask_dilation=source_mask_dilation,
        bad_amp=bad_amp,
        bad_amplifiers=bad_amplifiers,
    )
    print("\n >>> Finished reductions!\n")
    return {
        "records": records,
        "failures": failures,
        "science_count": len(science_records),
        "applied_modes": applied_modes,
        "manifest": manifest,
        "qa": qa_rows,
        "bad_amplifiers": bad_amplifiers,
        "output_directory": output_root,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".", help="Directory of raw FITS data")
    parser.add_argument(
        "--flat-mode",
        choices=("lamp", "superflat"),
        default="lamp",
        help="Flat construction mode (default: lamp)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output root containing cals/ and data/ (default: DATA/reduced)",
    )
    parser.add_argument(
        "--bad-amp",
        choices=BAD_AMP_CHOICES,
        default="auto",
        help="Bad Quad amplifier handling (default: auto)",
    )
    parser.add_argument("--boresight-separation-arcsec", type=float, default=15.0)
    parser.add_argument("--min-superflat-boresights", type=int, default=4)
    parser.add_argument("--source-mask-sigma", type=float, default=3.0)
    parser.add_argument("--source-mask-dilation", type=int, default=15)
    parser.add_argument("--illumination-smoothing-sigma", type=float, default=64.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_pipeline(
        directory=args.directory,
        flat_mode=args.flat_mode,
        output_directory=args.output_dir,
        boresight_separation_arcsec=args.boresight_separation_arcsec,
        min_superflat_boresights=args.min_superflat_boresights,
        source_mask_sigma=args.source_mask_sigma,
        source_mask_dilation=args.source_mask_dilation,
        illumination_smoothing_sigma=args.illumination_smoothing_sigma,
        bad_amp=args.bad_amp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
