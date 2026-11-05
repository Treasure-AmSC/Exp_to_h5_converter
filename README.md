# Exp_to_h5_converter
conversion from ATLAS/CMS root forma to h5


Phase 0: Data preparation
  ROOT (PHYSLITE) -> H5 (flat arrays)
## Files

| File | Purpose |
|------|---------|
| **Data preparation** | |
| `download_and_convert.py` | Download ATLAS open data ROOT files, convert to H5 |
| `atlas_to_h5_converter.py` | Library: ROOT to H5 conversion (imported) |
## Phase 0: Data Preparation

### Step 1: Download and convert ROOT to H5

```bash
# Download sample files from CERN Open Data and convert
python download_and_convert.py --download-samples --output-dir ./data --delete-root
```

**Convert from local ROOT files:**

```bash
python download_and_convert.py \
  --signal-files signal.root \
  --background-files bg.root \
  --data-files data.root \
  --output-dir ./data
```

**Convert from Rucio datasets (remote `root://` files, no download):**

```bash
# By default all files are used. Use --max-*-files to limit per category.
# Uses XCache server: root://xrootd03.usatlas.bnl.gov:1094//atlas/rucio
python download_and_convert.py \
  --signal-dataset "mc20_13TeV:signal_dataset" \
  --background-dataset "mc20_13TeV:bg_dataset" \
  --data-dataset "data15_13TeV:data_dataset" \
  --max-signal-files 10 --max-background-files 50 \
  --output-dir ./data
```

**Use direct XRootD server (no cache, BNL-OSG2_LOCALGROUPDISK replicas):**

```bash
# Uses root://dcgftp.usatlas.bnl.gov:1096/ directly, bypassing XCache
python download_and_convert.py \
  --signal-dataset "mc20_13TeV:signal_dataset" \
  --background-dataset "mc20_13TeV:bg_dataset" \
  --no-cache \
  --output-dir ./data
```

H5 files are labelled as `simulation` or `data` 

## CMS ROOT to H5

Run this with `uproot`, `awkward`, `numpy` and `h5py` available in your Python
environment:

```bash
python cms_nanoaod_to_h5_converter.py input.root -o output.h5
```

The converter uses the five collections' CMS study features and object
cuts, with the study event-cleaning flags applied to MC. The original
padded layout (up to 20 objects per collection), metadata and weight
sums are unchanged. PF candidates are ignored even when present; tracks
and generator particles are not exported. There is no benchmark-specific
event selection in the converter.

The existing electron, muon and jet pT options default to the study
thresholds of 10, 10 and 20 GeV. As before, an existing output file is overwritten.

### Earlier review sample files

- `CMS_Tau_data_pfnano.h5`: converted one complete Tau PFNano ROOT file from
  [CMS record 31315](https://opendata.cern.ch/record/31315).
- `CMS_TTTo2L2Nu_mc_nanoaod.h5`: converted one complete TTTo2L2Nu NanoAOD ROOT
  file from [CMS record 67801](https://opendata.cern.ch/record/67801).
- `ATLAS_data_DAOD_PHYSLITE.37019990._000001.pool.root.1.h5`: copied from our
  existing ATLAS H5 collection.
- `ATLAS_mc_DAOD_PHYSLITE.37620644._000012.pool.root.h5`: ttbar, DSID 410470.

These samples were produced with the earlier review converter. Its Tau
PFNano output included PF candidates; the TTTo2L2Nu output had empty track arrays.

