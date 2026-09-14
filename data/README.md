# Data layout

Raw survey data and generated model inputs are not committed to this Git
repository. Follow the survey-specific layouts below, then let the preparation
scripts write derived files to `data/processed/`.

## CRTS

The official Catalina Surveys Southern Periodic Variable Catalog download page
is:

http://nesssi.cacr.caltech.edu/DataRelease/VarcatS.html

Download the numerical catalogue and the associated CSDR2 photometry archive:

```bash
mkdir -p data

curl -L \
  http://nesssi.cacr.caltech.edu/DataRelease/SSS_Per_Tab.dat \
  -o data/SSS_Per_Tab.dat

curl -L \
  http://nesssi.cacr.caltech.edu/DataRelease/SSS_Per_Var_Cat.tar.gz \
  -o data/SSS_Per_Var_Cat.tar.gz

tar -xzf data/SSS_Per_Var_Cat.tar.gz -C data
python data/prepare_crts_layout.py
```

The official archive extracts to the flat directory
`data/SSS_Per_Var_Cat/`. The layout utility reads the numerical type from
`SSS_Per_Tab.dat` and creates hard links under the class-specific directories
required by the experiment code. It includes catalogue types 1–10 and 12;
type 11 (miscellaneous) and type 13 (LMC classical Cepheids) are outside the
inherited 11-class benchmark. Use `--mode copy` if the filesystem does not
support hard links.

After this step, the CRTS preparation script expects and finds:

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
The layout step preserves the downloaded files and their object identifiers;
it only makes the class directories expected by the preparation code.

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
