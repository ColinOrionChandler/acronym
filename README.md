## Automatic ARCTIC reductions
[![JOSS](http://joss.theoj.org/papers/a3d09040637afbcff0a01e0d77c563cf/status.svg)](http://joss.theoj.org/papers/a3d09040637afbcff0a01e0d77c563cf)
[![DOI](https://zenodo.org/badge/56095291.svg)](https://zenodo.org/badge/latestdoi/56095291) 

Kolby L. Weisenburger, Joseph Huehnerhoff, Emily M. Levesque, Philip Massey

2017

**Acronym** is an automatic reduction pipeline for the Astrophysical Research Consortium Telescope Imaging Camera ([ARCTIC](http://www.apo.nmsu.edu/arc35m/Instruments/ARCTIC/)) at Apache Point Observatory ([APO](http://www.apo.nmsu.edu/)). Despite an increasing number of telescopes and observatories coming online, instrument-specific image reduction packages have been severely lacking. Historically, astronomers have resorted to building in-house (potentially ad-hoc) software to calibrate raw astronomical images; these different reduction algorithms can lead to discrepant scientific results. **Acronym** aims to streamline this image reduction process such that all ARCTIC users can benefit from 1. an open-source and open-access automatic reduction pipeline and 2. a normalized tool so that they are able to compare apples to apples and galaxies to galaxies. 

We developed in-house procedures to reduce ARCTIC images rather than using other similar packages (e.g. [astropy's ccdproc](https://github.com/astropy/ccdproc), [DOI: 10.5281/zenodo.47652](https://zenodo.org/record/47652)) to handle ARCTIC's various CCD readout modes. While these unique readout modes could have been enveloped into a helper function and used in tandem with astropy's ccdproc, we wished to maintain a package that was curated specifically for the ARCTIC instrument and that did not rely on preexisting reduction packages.

<img src="https://github.com/kweis/acronym/blob/master/docs/Aligned_m106.png" width="500" height="500" align="middle"/>

_Stacked M106 image using reduced Johnson V, R, and I band images (reduced by **acronym**)._


To use:

**python acronym.py [your directory of data]**

OR place .py in your folder with data and run with no argument:

**python acronym.py**

Lamp flats remain the default, so existing commands keep the same interface. To
build a science-assisted superflat instead:

```bash
python acronym.py /path/to/raw/data --flat-mode superflat
```

Use `--output-dir` to keep lamp and superflat trials separate:

```bash
python acronym.py /path/to/raw/data \
  --flat-mode lamp \
  --output-dir /path/to/raw/data/reduced_lamp

python acronym.py /path/to/raw/data \
  --flat-mode superflat \
  --output-dir /path/to/raw/data/reduced_superflat
```

## Science-assisted superflats

Superflat mode first builds the normal high-signal-to-noise lamp flat. Science
frames are then grouped by filter and boresight using their `RA` and `DEC`
headers. Pointings whose complete-linkage separation is no more than 15 arcsec
belong to one boresight. One source-masked frame is selected from each boresight
so repeated visits to a field do not dominate the result.

For filters with at least four usable boresights, Acronym writes:

- `master_lamp_flat_FILTER.fits`: the lamp-only response;
- `master_science_flat_FILTER.fits`: a full, diagnostic science-derived flat;
- `illumination_correction_FILTER.fits`: the smoothed science residual;
- `master_superflat_FILTER.fits`: lamp flat times illumination correction;
- `superflat_coverage_FILTER.fits`: the number of usable inputs per pixel; and
- `master_flat_FILTER.fits`: the flat actually applied (the hybrid product).

If fewer than four usable boresights are available, the applied
`master_flat_FILTER.fits` remains the lamp flat and its FITS header records
`FLATMODE=lamp-fallback`. `superflat_manifest.csv` records grouping, selection,
masking, and fallback decisions. `calibration_qa.csv` records source-masked
background uniformity metrics for every reduced science frame.

The main tunable options are:

```text
--boresight-separation-arcsec 15
--min-superflat-boresights 4
--source-mask-sigma 3
--source-mask-dilation 15
--illumination-smoothing-sigma 64
```

All calibration modes apply per-amplifier overscan correction, subtract a
master bias from darks/flats/science frames, and prefer an exact-exposure master
dark. When an exact dark is unavailable, the longest available master dark is
scaled to the requested exposure.

Note that there is a requirements.txt file. To install necessary dependencies:

**pip install -r requirements.txt**

## Example

You can test the **acronym** pipeline using the rawdata zipfile (example.zip in [release docs](https://github.com/kweis/acronym/releases/tag/v1.0.1)) or an ARCTIC dataset of your choosing. 


Expected output for example directory reduction:

  ```bash
04:51:10 $ python acronym.py example/rawdata

 >>> Starting bias combine...
   > Created master bias

 >>> Starting darks...
   > Created master 10.0 second dark
   > Created master 30.0 second dark
   > Created master 5.0 second dark
   > Created master 60.0 second dark
   > No darks found for exposure time 120.0 sec. Continuing reductions...

 >>> Starting flats...
   > Created master MSSSO R flat
   > Created master MSSSO V flat

 >>> 7 science images found. Starting reductions...

 >>> Finished reductions! 
  ```
  
  This created _example/rawdata/reduced/cals/_ and _example/rawdata/reduced/data/_. In case of missing select calibration (e.g. 30 second darks), **acronym** will alert you, but continue to reduce the other images. 
  

Please contact Kolby Weisenburger (kweis@uw.edu) with questions, issues or contributions.
