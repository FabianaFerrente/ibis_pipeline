# Optics & Polarization Modeling

Simulation and calibration of the telescope optics via Mueller matrix formalism.

## Modules

- `mueller_matrix.py` – T-matrix construction (constructTMatrix)
- `optical_elements.py` – Mirror, retarder, polarizer functions (mirror, retarder)
- `angle_geometry.py` – Normalize angles and apply 2D/4D rotations
- `ibis_tmtx.py` – Wrapper to compute IBIS Mueller matrices

## Notes

This module is essential for accurate polarimetric calibration.
