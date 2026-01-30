"""Train RidgeCV meta-learner on pooled OOF from scripts 3 and 4 (real OOF parquets)."""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_stacking import train_stacking


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    path_a = out / "oof_model_a.parquet"
    path_b = out / "oof_model_b.parquet"
    if not path_a.exists() or not path_b.exists():
        print(
            "OOF parquets not found. Run scripts 3 and 4 with OOF output first "
            "(outputs/oof_model_a.parquet and outputs/oof_model_b.parquet).",
            file=sys.stderr,
        )
        sys.exit(1)
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    merged = df_a.merge(
        df_b,
        on=["team_id", "as_of_date"],
        how="inner",
        suffixes=("", "_b"),
    )
    if "y_b" in merged.columns:
        merged = merged.drop(columns=["y_b"])
    if merged.empty:
        print("No overlapping (team_id, as_of_date) between OOF files.", file=sys.stderr)
        sys.exit(1)
    # Impute NaN in OOF columns (e.g. Model A numerical instability) so RidgeCV gets finite X
    for col in ["oof_a", "oof_xgb", "oof_rf", "y"]:
        if col in merged.columns and merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].mean())
    oof_a = merged["oof_a"].values.astype("float32")
    oof_xgb = merged["oof_xgb"].values.astype("float32")
    oof_rf = merged["oof_rf"].values.astype("float32")
    y = merged["y"].values.astype("float32")
    path = train_stacking(oof_a, oof_xgb, oof_rf, y, config, out)
    print(f"Saved {path}, {out / 'oof_pooled.parquet'}")


if __name__ == "__main__":
    main()
