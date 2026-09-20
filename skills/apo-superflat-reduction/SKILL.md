---
name: apo-superflat-reduction
description: "Redo APO ARCTIC data reductions with superflats from an existing reduced-data path, including Dropbox hydration checks, object organization, local plate solving, object-centered cutouts, and phase manifests. Use for requests such as redo this reduction with superflats."
---

# APO superflat reduction

Use the maintained runner at `/Users/colinchandler/GitHub/acronym/redo_superflat.py`
with `/Users/colinchandler/opt/anaconda3/envs/COC/bin/python`. Its maintained skill
source is `/Users/colinchandler/GitHub/acronym/skills/apo-superflat-reduction`;
keep this installed copy synchronized when changing the workflow.

## Execute

1. Inspect the supplied reduced path and the runner's `--help`. A request to redo
   the reduction authorizes executing the complete workflow. Preserve raw FITS
   and the existing reduction. Resolve filesystem access through the tool's
   normal permissions mechanism when the output is outside writable roots.
2. Invoke `python redo_superflat.py REDUCED_PATH`, supplying `--master MASTER_PATH`
   if the user provided one. The master can be the `UTyymmdd` directory or its
   `arctic` child. Quote paths; do not retain presentation escapes such as `\_`.
   `--preflight-only` inspects without creating the output. Normal execution
   includes preflight, so a separate full preflight is optional.
3. Default output is `REDUCED_PATH.parent/superflat_processed`. Keep per-phase
   logs and checkpoints there. Use `--resume` only with the same input paths and
   settings; the runner verifies saved source hashes and completed artifacts.
   Do not overwrite or discard a conflicting output tree. A stale lock requires
   confirming the recorded PID is no longer running before removing it.
4. Monitor the process and logs without starting duplicate reductions. If a
   failure can be diagnosed and safely corrected within the authorized task,
   correct it and resume. Otherwise preserve progress, report the specific
   problem, and ask what to do. Do not relax scientific thresholds to force
   success.

## Scientific and operational contracts

- Include all raw science exposures, even those missing from the old reduction.
  Exclude everything beneath a directory named `bad`, including calibration
  inputs, and record exclusions. The current reducer accepts top-level `.fits`
  files; unfamiliar layouts require deliberate staging, not silent omissions.
- Verify complete readable image payloads and Dropbox offline/placeholder
  indicators, not merely nonzero file sizes. Empty ancillary metadata does not
  imply missing science data. Bounded reads attempt hydration before pausing.
  Nonstandard `+NAN` WCS cards in ARCTIC calibration headers are not evidence of
  truncated image payloads; do not modify raw headers to normalize them.
- Retain acronym's superflat defaults, minimum four usable boresights, lamp
  fallback, and automatic amplifier detection. Do not assume a failed amplifier
  from another observing night applies here.
- Preserve `cals/` and `data/`; object-organized copies are placed directly in
  `<object>/ARCTIC/`. Replace slashes with `+` and spaces with `_`; unresolved
  names and collisions require attention.
- Use the installed local astrometry.net `solve-field` and accessible local
  indexes. Never substitute an online plate-solving service. A solved marker
  and validated celestial WCS are required for success. Solving changes only
  derived WCS headers, never science pixels.
- If more than half the science exposures solve, finish cutouts for the run.
  Failed solves may use usable original WCS, explicitly marked unverified.
  Record missing-WCS and out-of-image failures; never fabricate centered
  products. If half or fewer solve, pause and ask.
- Center 126-arcsecond cutouts on Horizons positions at exposure midpoint,
  observatory `705`. Cache exact query identifiers, epochs, coordinates, and
  resolved names in the run. For ambiguous names, inspect returned candidates
  and resolve from evidence; ask if still uncertain. Use `--target-map FILE`
  with a JSON name-to-Horizons-ID mapping to resume unresolved targets; changes
  to targets that already produced ephemerides are refused.
- Retain native detector pixels and NaNs. FITS/PNG/arrow-PDF triplets go under
  `<object>/ARCTIC/cutouts/`, with a fixed 551x551 pixel size for every cutout.
  Use lossless quarter-turns/flips when the WCS supports them; retain native
  orientation for distortion models that cannot be safely transformed. Record
  `CUTMODE=D4-PIX`, `CUTRSMP=False`, transform, and WCS source. These are
  predicted-position cutouts, not claims of detected objects.
- Name each FITS, PNG, and arrow PDF with
  `<object>_<YYYYMMDD_HHMMSS>_<band>_<exptime>s_<thumbnail>_126arcsec_NuEl`.
  The timestamp is the exposure midpoint rounded to the nearest second, the
  band is a readable label such as `red`, and `<thumbnail>` retains the source
  image identity. For objects with at least two cutout PNGs, create one GIF
  using the same stem as the first chronological cutout (so its filename has
  the first datetime stamp), ordered by exposure midpoint, with the 551x551 PNG
  dimensions unchanged, 250 ms per frame, and an infinite loop. If an object has
  only one PNG, skip GIF creation and record that decision in the manifests.

## Verify and report

The runner maintains `master_manifest.csv` (one row per object),
`frame_manifest.csv` (input/exclusion and per-exposure status),
`pipeline_state.json` (resume state and artifact/source hashes), and
`validation.json` (final audit). Preserve acronym's `cals/calibration_qa.csv`
and `cals/superflat_manifest.csv`.

Require raw/reduced input preservation, exposure/QA/manifest parity, validated
WCS and target containment, pixel-preserving cutouts, and nonempty readable
outputs. Inspect representative rendered PNGs, arrow PDFs, and GIFs before
reporting completion. Report actual hybrid versus lamp-fallback counts, solved
versus unverified-WCS cutouts, GIF frame dimensions/timing, missing products,
and explicit unresolved exceptions.
Link the output root and master manifest in the final response.
