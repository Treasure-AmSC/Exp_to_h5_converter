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

