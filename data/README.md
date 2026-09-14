# Data layout

Raw survey data and generated model inputs are not committed to this Git
repository. Place downloaded files under `data/raw/`, then let the preparation
scripts write derived files to `data/processed/`.

## CRTS

The CRTS preparation script expects the following layout:

```text
data/
├── SSS_Per_Tab.dat
└── cartlinDR2/
    └── original_data/
        └── type/
            ├── 1/*.dat
            ├── 2/*.dat
            └── ...
```

`SSS_Per_Tab.dat` contains catalogue metadata. Each light-curve file is a
whitespace-separated table consumed by `experiments/crts/1_prepare_crts_tdmn.py`.
The 11-class benchmark follows the CRTS/CSDR2 benchmark construction described
in the accompanying paper. Preserve the original object identifiers and class
directories.

Catalogue reference:

> Drake, A. J., et al. (2017). The Catalina Surveys Southern periodic variable
> star catalogue. *MNRAS*, 469, 3688–3712.
> https://doi.org/10.1093/mnras/stx1085

## OGLE-IV

Download the public OGLE Collection of Variable Stars files from:

https://www.astrouw.edu.pl/ogle/ogle4/OCVS/

The preparation script expects this structure below `data/raw/ogle4/`:

```text
data/raw/ogle4/
├── lmc/
│   ├── acep/{ident.dat,phot/}
│   ├── cep/{ident.dat,phot/}
│   ├── dsct/{ident.dat,phot/}
│   ├── ecl/{ident.dat,phot/}
│   ├── rrlyr/{ident.dat,phot/}
│   └── t2cep/{ident.dat,phot/}
└── smc/
    └── ... same class directories ...
```

Within each class directory, the script reads catalogue `.dat` files and
photometry from `phot/I/<OGLE_ID>.dat`, falling back to the V-band path when
needed. Do not rename catalogue IDs.

## Generated files

The default output paths are:

```text
data/processed/crts_tdmn.h5
data/processed/crts_tdmn.manifest.csv
data/processed/crts_upsilon_16_features.csv
data/processed/ogle_tdmn.h5
data/processed/ogle_tdmn.manifest.csv
data/processed/ogle_upsilon_16_features.csv
```

The manifests contain local `source_path` values so that UPSILoN feature
extraction can reopen the same raw light curves. If a prepared dataset is moved
to another machine, regenerate the manifest or update only this path column;
do not change object IDs, split labels, or class labels.

## Publishing processed data

Generated HDF5 files may exceed GitHub's file-size limits. Publish them through
a research-data repository such as Zenodo or an institutional repository, then
add the permanent DOI/download URL to the top-level README and to the article's
Data Availability Statement. Include SHA-256 checksums for every released HDF5
and manifest file.

