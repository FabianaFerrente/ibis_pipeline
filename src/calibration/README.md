# Calibration Modules

Core calibration routines for spectroscopic and spectropolarimetric data, including flat-fielding, alignment, wavelength shift correction, and gain normalization.

## Modules

- `alignment_nb_bb.py` – Align Narrowband/Broadband data
- `dark_flat_nb_bb.py` – Flat-field and dark correction
- `gain.py` – Gain calibration
- `systematic_wavelengthshift.py` – Correction of wavelength shifts across the field
- `polarisation_calibration_curves.py` – Build Mueller matrices from calibration curves
- `main_spectroscopic_calibration.py` – Spectroscopic calibration workflow
- `main_spectropolarimetric_calibration.py` – Full polarimetric calibration workflow
- `preparation_momfbd.py` – MOMFBD preparation and data reformatting

## Notes

Each module should ideally be independently testable and reusable.
