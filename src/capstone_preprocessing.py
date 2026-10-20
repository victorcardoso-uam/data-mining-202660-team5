"""
IIND4417 — Capstone Milestone 2, Team 5 (Big99 Inventory Intelligence)
src/capstone_preprocessing.py

Turns the confidential Big99 inventory export into a clean, anonymised
modelling table. Every column that is only known AFTER a unit has aged in the
warehouse is dropped here, so no downstream model can see the answer.

    data/raw/inventario_mfp_usado.csv        (confidential, git-ignored)
        -> data/processed/big99_model_input.csv   (anonymised, committed)

From the project root:
    python src/capstone_preprocessing.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "inventario_mfp_usado.csv"
OUT_PATH = ROOT / "data" / "processed" / "big99_model_input.csv"

SLOW_MOVER_DAYS = 120

# Purchase-time attributes: known the day the unit is bought.
RENAME = {
    "MARCA": "brand",
    "LINEA": "line",
    "UBICACION": "warehouse",
    "APARIENCIA GRAL.": "appearance",
    "CALIDAD COPIA": "copy_quality",
    "EQUIPO FUNCIONANDO": "working",
    "CONTADOR": "page_counter_k",
    "COST W/FREIGHT": "landed_cost_mxn",
}

ACCESSORY_TOKENS = ["DUPLEX", "DSDF", "FAX", "RADF", "FINISHER", "U-SEND"]

# Columns removed on purpose, with the reason printed in the execution log.
LEAKAGE_COLUMNS = {
    "DIAS EN INVENTARIO": "target source (kept only to build the label)",
    "RANGO": "age bucket = the target in disguise",
    "FECHA LLEGADA": "arrival date fully determines days in stock",
    "T.C.": "exchange rate at purchase acts as a date proxy",
    "APARTADO": "reservation happens after the unit starts ageing",
    "ULTIMO MOVIMIENTO": "transfers accumulate as the unit ages",
    "MOVIMIENTOS INTERNOS": "post-arrival movement history",
    "ES": "current stock status flag",
}
CONFIDENTIAL_COLUMNS = [
    "CLIENTE", "VENDEDOR", "PROVEEDOR", "FOLIO FACTURA PROVEEDOR",
    "PEDIMENTO", "SERIE", "COMENTARIOS", "OBSERVACIONES",
]


def section(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def main():
    print("=" * 78)
    print("BIG99 CAPSTONE — PREPROCESSING & LEAKAGE AUDIT")
    print("=" * 78)
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    print(f"Raw export: {raw.shape[0]} rows x {raw.shape[1]} columns")

    section("1. INVALID RECORDS")
    invalid = (raw["COST W/FREIGHT"] <= 0) | (raw["DIAS EN INVENTARIO"] <= 0)
    print(f"Rows with zero landed cost or zero days in stock: {int(invalid.sum())}")
    print(raw.loc[invalid, ["MARCA", "COST W/FREIGHT", "DIAS EN INVENTARIO",
                            "ULTIMO MOVIMIENTO"]].to_string())
    df = raw.loc[~invalid].copy()
    print(f"-> dropped, {len(df)} rows remain")

    section("2. MISSING VALUES")
    counter = pd.to_numeric(df["CONTADOR"], errors="coerce")
    zero_counter = int((counter == 0).sum())
    df["CONTADOR"] = counter.replace(0, np.nan)
    print(f"page counter = 0 on a used MFP is a missing reading: {zero_counter} "
          "-> set to NaN")
    print("Imputation (median) is NOT done here: it is fitted inside the model "
          "pipeline on training folds only, so test rows never inform it.")

    section("3. LEAKAGE & CONFIDENTIALITY FILTER")
    for column, reason in LEAKAGE_COLUMNS.items():
        print(f"  drop {column:<22} {reason}")
    print(f"  drop {len(CONFIDENTIAL_COLUMNS)} confidential columns "
          "(client, seller, supplier, invoice, customs, serial, free text)")

    section("4. FEATURE CONSTRUCTION")
    accessories = df["ACCESORIOS"].astype(str).str.upper()
    for token in ACCESSORY_TOKENS:
        df["acc_" + token.lower().replace("-", "")] = (
            accessories.str.contains(token, regex=False).astype(int)
        )
    print(f"ACCESORIOS free text -> {len(ACCESSORY_TOKENS)} binary flags "
          f"({', '.join(ACCESSORY_TOKENS)}); 'FINISHER' merges 30+ model variants")

    # Units of one purchase lot share arrival date and therefore label.
    # An anonymous lot id lets the model split by lot instead of by row.
    lot_key = df["PEDIMENTO"].fillna(df["FECHA LLEGADA"]).astype(str)
    df["lot_id"] = "LOT_" + pd.Series(
        pd.factorize(lot_key)[0] + 1, index=df.index).astype(str).str.zfill(3)
    print(f"Purchase lots identified: {df['lot_id'].nunique()} "
          "(anonymised, used for grouped validation)")

    df["slow_mover"] = (df["DIAS EN INVENTARIO"] > SLOW_MOVER_DAYS).astype(int)
    df = df.rename(columns=RENAME)

    columns = (["lot_id"] + list(RENAME.values())
               + [c for c in df.columns if c.startswith("acc_")]
               + ["slow_mover"])
    out = df[columns]

    section("5. OUTPUT")
    print(f"Modelling table: {out.shape[0]} rows x {out.shape[1]} columns")
    print(f"slow_mover = days in stock > {SLOW_MOVER_DAYS}: "
          f"{out['slow_mover'].sum()} positives ({out['slow_mover'].mean():.1%})")
    print(f"Missing values left for the pipeline: "
          f"{out.isna().sum()[out.isna().sum() > 0].to_dict()}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH.relative_to(ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
