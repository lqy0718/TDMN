#!/usr/bin/env python3
"""Arrange the official flat CRTS light-curve archive by catalogue type.

The accepted-paper preparation script reads class-specific directories.  The
official ``SSS_Per_Var_Cat.tar.gz`` archive is flat, so this utility creates the
required directory layout without changing any light-curve content.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = DATA_ROOT / "SSS_Per_Tab.dat"
DEFAULT_LIGHT_CURVES = DATA_ROOT / "SSS_Per_Var_Cat"
DEFAULT_OUTPUT = DATA_ROOT / "cartlinDR2/original_data/type"
INCLUDED_TYPES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)


def load_catalog(path):
    return pd.read_csv(
        path,
        header=None,
        skiprows=3,
        sep=r"\s+",
        usecols=range(9),
        names=(
            "SSS_ID",
            "ID",
            "RA",
            "Dec",
            "Period",
            "V_CSS",
            "Npts",
            "V_amp",
            "Type",
        ),
        dtype={"ID": str, "Type": int},
    )


def place_file(source, destination, mode):
    if destination.exists() or destination.is_symlink():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Existing destination has a different size: {destination}"
            )
        return "existing"

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, start=destination.parent))
    else:
        try:
            os.link(source, destination)
        except OSError as exc:
            raise OSError(
                f"Could not hard-link {source} to {destination}. "
                "Rerun with --mode copy or --mode symlink."
            ) from exc
    return mode


def main():
    parser = argparse.ArgumentParser(
        description="Arrange official CRTS light curves into class directories."
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--light-curve-dir", default=str(DEFAULT_LIGHT_CURVES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="hardlink avoids duplicating the approximately 100 MB archive",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog).expanduser().resolve()
    light_curve_dir = Path(args.light_curve_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"CRTS catalogue not found: {catalog_path}")
    if not light_curve_dir.is_dir():
        raise FileNotFoundError(f"Extracted CRTS directory not found: {light_curve_dir}")

    catalog = load_catalog(catalog_path)
    selected = catalog[catalog["Type"].isin(INCLUDED_TYPES)].copy()
    if selected["ID"].duplicated().any():
        duplicated = selected.loc[selected["ID"].duplicated(), "ID"].head().tolist()
        raise ValueError(f"Duplicate numerical IDs in catalogue: {duplicated}")

    missing = []
    counts = {type_id: 0 for type_id in INCLUDED_TYPES}
    actions = {"hardlink": 0, "copy": 0, "symlink": 0, "existing": 0}
    for row in selected.itertuples(index=False):
        source = light_curve_dir / f"{row.ID}.dat"
        if not source.is_file():
            missing.append(str(source))
            continue
        destination_dir = output_root / str(row.Type)
        destination_dir.mkdir(parents=True, exist_ok=True)
        action = place_file(source, destination_dir / source.name, args.mode)
        actions[action] += 1
        counts[int(row.Type)] += 1

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} catalogue light curves are missing; examples: {missing[:10]}"
        )

    expected = selected.groupby("Type").size().to_dict()
    if any(counts[type_id] != int(expected.get(type_id, 0)) for type_id in INCLUDED_TYPES):
        raise RuntimeError(f"Organized counts do not match the catalogue: {counts}")

    print(f"CRTS layout ready: {output_root}")
    print(f"Actions: {actions}")
    print(f"Included type counts: {counts}")
    print(f"Total benchmark objects: {sum(counts.values())}")


if __name__ == "__main__":
    main()
