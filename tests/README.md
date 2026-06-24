# PFNano Support Tests

## Changes

`cms_nanoaod_to_h5_converter.py` now accepts PFNano input files in addition
to standard NanoAOD. When `PFCands` branches are present:

- **`common/tracks`** is filled from charged PFCands (pT > 0.5 GeV), matching
  the ATLAS track schema (pt, eta, phi, d0, z0).
- **`cms/pfcands`** stores all PFCands (charged + neutral, pT > 0.2 GeV) with
  pdgId, puppiWeight, track quality, etc.

When PFCands are absent, the converter behaves as before with standard NanoAOD.

`pfnano_validation.ipynb` runs checks against a PFNano ROOT input and its converted HDF5 output.

Running the notebook with a test PFNano ROOT file requires:

- Python 3.7+
- `uproot`, `awkward`, `numpy`, `h5py`