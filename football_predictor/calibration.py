"""Calibracion simple de probabilidades usando resultados de backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BUCKET_SIZE = 0.05
MIN_BUCKET_SAMPLES = 8


@dataclass(frozen=True)
class CalibrationResult:
    market: str
    raw_probability: float
    calibrated_probability: float
    bucket: str | None
    samples: int
    hit_rate: float | None

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def build_calibration_table(backtest_details: pd.DataFrame, bucket_size: float = BUCKET_SIZE) -> pd.DataFrame:
    """Construye tabla de calibracion por mercado y bucket de probabilidad."""

    if backtest_details.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    market_specs = [
        ("1X2", "recommended_probability", "hit_1x2"),
        ("Over 2.5", "over25_prob", "hit_over25"),
        ("BTTS", "btts_yes_prob", "hit_btts"),
    ]
    if "recommended_market" in backtest_details.columns:
        market_specs.append(("Pick recomendado", "recommended_probability", "hit_recommended"))
        details = backtest_details.copy()
        details["hit_recommended"] = details.apply(_recommended_hit, axis=1)
    else:
        details = backtest_details

    for market, probability_column, hit_column in market_specs:
        if probability_column not in details.columns or hit_column not in details.columns:
            continue
        temp = details[[probability_column, hit_column]].dropna().copy()
        if temp.empty:
            continue
        temp["bucket_floor"] = (temp[probability_column].astype(float) // bucket_size) * bucket_size
        grouped = temp.groupby("bucket_floor")
        for bucket_floor, group in grouped:
            bucket_start = round(float(bucket_floor), 2)
            bucket_end = round(min(1.0, bucket_start + bucket_size), 2)
            rows.append(
                {
                    "market": market,
                    "bucket_start": bucket_start,
                    "bucket_end": bucket_end,
                    "bucket": f"{bucket_start:.0%}-{bucket_end:.0%}",
                    "samples": int(len(group)),
                    "avg_probability": round(float(group[probability_column].mean()), 4),
                    "hit_rate": round(float(group[hit_column].mean()), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "bucket_start"]).reset_index(drop=True) if rows else pd.DataFrame()


def apply_calibration(
    predictions: pd.DataFrame,
    calibration_table: pd.DataFrame,
    min_samples: int = MIN_BUCKET_SAMPLES,
) -> pd.DataFrame:
    """Agrega probabilidad calibrada para el pick recomendado."""

    if predictions.empty:
        return predictions
    calibrated = predictions.copy()
    results = []
    for _, row in calibrated.iterrows():
        result = calibrate_probability(
            raw_probability=float(row.get("recommended_probability", 0.0) or 0.0),
            market="Pick recomendado",
            calibration_table=calibration_table,
            min_samples=min_samples,
        )
        results.append(result.as_dict())
    result_df = pd.DataFrame(results).rename(
        columns={
            "calibrated_probability": "calibrated_pick_probability",
            "bucket": "calibration_bucket",
            "samples": "calibration_samples",
            "hit_rate": "calibration_hit_rate",
        }
    )
    keep = ["calibrated_pick_probability", "calibration_bucket", "calibration_samples", "calibration_hit_rate"]
    calibrated = pd.concat([calibrated.reset_index(drop=True), result_df[keep].reset_index(drop=True)], axis=1)
    return calibrated


def calibrate_probability(
    raw_probability: float,
    market: str,
    calibration_table: pd.DataFrame,
    min_samples: int = MIN_BUCKET_SAMPLES,
) -> CalibrationResult:
    if calibration_table.empty:
        return CalibrationResult(market, raw_probability, raw_probability, None, 0, None)

    bucket_floor = (raw_probability // BUCKET_SIZE) * BUCKET_SIZE
    candidates = calibration_table[
        (calibration_table["market"] == market)
        & (calibration_table["bucket_start"] <= raw_probability)
        & (raw_probability < calibration_table["bucket_end"])
        & (calibration_table["samples"] >= min_samples)
    ]
    if candidates.empty:
        return CalibrationResult(market, raw_probability, raw_probability, None, 0, None)

    row = candidates.iloc[0]
    hit_rate = float(row["hit_rate"])
    # Mezcla conservadora: no reemplaza totalmente el modelo cuando el bucket es pequeno.
    samples = int(row["samples"])
    weight = min(0.75, samples / 80)
    calibrated = (raw_probability * (1 - weight)) + (hit_rate * weight)
    return CalibrationResult(
        market=market,
        raw_probability=round(raw_probability, 4),
        calibrated_probability=round(float(calibrated), 4),
        bucket=str(row["bucket"]),
        samples=samples,
        hit_rate=round(hit_rate, 4),
    )


def summarise_markets(backtest_details: pd.DataFrame) -> pd.DataFrame:
    """Resume rendimiento por mercado recomendado y accion."""

    if backtest_details.empty or "recommended_market" not in backtest_details.columns:
        return pd.DataFrame()
    details = backtest_details.copy()
    details["hit_recommended"] = details.apply(_recommended_hit, axis=1)
    group_columns = ["recommended_market"]
    if "action" in details.columns:
        group_columns.append("action")
    summary = (
        details.groupby(group_columns, dropna=False)
        .agg(
            Partidos=("recommended_market", "size"),
            Acierto=("hit_recommended", "mean"),
            Confianza=("confidence_score", "mean"),
            Probabilidad=("recommended_probability", "mean"),
        )
        .reset_index()
        .sort_values(["Acierto", "Partidos"], ascending=[False, False])
    )
    return summary.round(4)


def _recommended_hit(row: pd.Series) -> int:
    market = str(row.get("recommended_market", ""))
    if market == "Gana local":
        return int(row.get("actual_result") == "H")
    if market == "Empate":
        return int(row.get("actual_result") == "D")
    if market == "Gana visitante":
        return int(row.get("actual_result") == "A")
    if market == "Doble oportunidad 1X":
        return int(row.get("actual_result") in {"H", "D"})
    if market == "Doble oportunidad X2":
        return int(row.get("actual_result") in {"D", "A"})
    if market == "Doble oportunidad 12":
        return int(row.get("actual_result") in {"H", "A"})
    if market == "Over 1.5":
        return int(float(row.get("actual_total_goals", 0)) > 1.5)
    if market == "Under 1.5":
        return int(float(row.get("actual_total_goals", 0)) < 1.5)
    if market == "Over 2.5":
        return int(float(row.get("actual_total_goals", 0)) > 2.5)
    if market == "Under 2.5":
        return int(float(row.get("actual_total_goals", 0)) < 2.5)
    if market == "Over 3.5":
        return int(float(row.get("actual_total_goals", 0)) > 3.5)
    if market == "Under 3.5":
        return int(float(row.get("actual_total_goals", 0)) < 3.5)
    if market == "Ambos anotan: Sí":
        return int(row.get("actual_btts") == 1)
    if market == "Ambos anotan: No":
        return int(row.get("actual_btts") == 0)
    return 0
