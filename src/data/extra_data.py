"""External & Extended Data Fetcher for Turku Energia Imbalance Price Challenge.

Supports two Q&A-sanctioned extensions without modifying official build_dataset.py:
  1. Pre-June 2026 historical aFRR & mFRR price/volume/direction series from Fingrid
     (used strictly as lag buffers and physical regulation features; target imbalance_price
      training is gated to >= 2026-06-01 due to the volume-based pricing methodology switch).
  2. Fingrid 3-minute real-time open-data series (181 Wind RT, 188 Nuclear RT,
     191 Production RT, 177 Power System State/Consumption RT), resampled to 15-minute
     delivery intervals with a configurable safety lag (default: 30 minutes to match
     official Fingrid measurement series in availability.csv; configurable to 10 minutes
     if approved by the hackathon jury).

Output is written to `data/extra_inputs.parquet` in the exact 7-column schema:
  [series_id, source, dataset_id, timestamp_utc, value, unit, first_usable_origin_utc]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.loader import DEFAULT_DATA_DIR, read_long

UTC = timezone.utc
FINGRID_API_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"
VOLUME_PRICING_REGIME_START_UTC = pd.Timestamp("2026-05-31T21:00:00Z")


@dataclass(frozen=True)
class ExtraFingridSeries:
    series_id: str
    dataset_id: int
    unit: str
    default_lag_minutes: int
    is_realtime_3min: bool = False
    note: str = ""


# 1. Real-time 3-minute Fingrid series highlighted in Hackathon Q&A
REALTIME_FINGRID_SERIES: list[ExtraFingridSeries] = [
    ExtraFingridSeries(
        series_id="fi_wind_rt",
        dataset_id=181,
        unit="MW",
        default_lag_minutes=30,
        is_realtime_3min=True,
        note="Wind power generation - real-time data (3-min resampled to 15-min)",
    ),
    ExtraFingridSeries(
        series_id="fi_nuclear_rt",
        dataset_id=188,
        unit="MW",
        default_lag_minutes=30,
        is_realtime_3min=True,
        note="Nuclear power generation - real-time data (3-min resampled to 15-min)",
    ),
    ExtraFingridSeries(
        series_id="fi_production_rt",
        dataset_id=191,
        unit="MW",
        default_lag_minutes=30,
        is_realtime_3min=True,
        note="Electricity production in Finland - real-time data (3-min resampled to 15-min)",
    ),
    ExtraFingridSeries(
        series_id="fi_grid_state_rt",
        dataset_id=177,
        unit="state",
        default_lag_minutes=30,
        is_realtime_3min=True,
        note="Power system state / real-time indicator (3-min resampled to 15-min)",
    ),
]

# 2. Historical pre-June 2026 aFRR & mFRR series (for 7-day lag buffer & regulation regime)
# Note: Pre-June fi_imbalance_price (319) is intentionally excluded per Hackathon Q&A
# because the imbalance price calculation methodology changed to volume-based in June 2026.
HISTORICAL_REGULATION_SERIES: list[ExtraFingridSeries] = [
    ExtraFingridSeries("fi_mfrr_up_price", 244, "EUR/MWh", 60, False, "mFRR up price pre-June"),
    ExtraFingridSeries("fi_mfrr_down_price", 106, "EUR/MWh", 60, False, "mFRR down price pre-June"),
    ExtraFingridSeries("fi_mfrr_up_activated", 375, "MW", 60, False, "mFRR up volume pre-June"),
    ExtraFingridSeries("fi_mfrr_down_activated", 376, "MW", 60, False, "mFRR down volume pre-June"),
    ExtraFingridSeries("fi_mfrr_direction", 369, "-1/0/1", 60, False, "mFRR direction pre-June"),
    ExtraFingridSeries("fi_afrr_up_price", 347, "EUR/MWh", 60, False, "aFRR up price pre-June"),
    ExtraFingridSeries("fi_afrr_down_price", 348, "EUR/MWh", 60, False, "aFRR down price pre-June"),
    ExtraFingridSeries("fi_consumption", 124, "MWh/h", 30, False, "Consumption pre-June"),
    ExtraFingridSeries("fi_production", 74, "MWh/h", 30, False, "Production pre-June"),
]


def compute_rt_first_usable(
    timestamp_utc: pd.Series | pd.DatetimeIndex,
    lag_minutes: int = 30,
    period_minutes: int = 15,
) -> pd.Series:
    """Compute leak-free `first_usable_origin_utc` for resampled quarter-hour measurements.

    Follows RULE #1 (`first_usable_origin_utc = delivery_start_utc + period_minutes + lag_minutes`).
    Defaults to lag_minutes=30 (matching Fingrid 124/74 in availability.csv) unless overridden
    after official jury confirmation (e.g. lag_minutes=10).
    """
    ts = pd.to_datetime(timestamp_utc, utc=True)
    total_offset = pd.Timedelta(minutes=period_minutes + int(lag_minutes))
    return pd.Series(ts + total_offset, index=getattr(timestamp_utc, "index", None))


def resample_3min_to_quarter_hour(
    raw_df: pd.DataFrame,
    spec: ExtraFingridSeries,
    rt_lag_minutes: int = 30,
) -> pd.DataFrame:
    """Aggregate 3-minute Fingrid real-time observations into 15-minute quarter-hour buckets.

    Each 3-minute measurement inside [T, T+15min) is averaged into the quarter-hour
    starting at `T`, and `first_usable_origin_utc` is set to `T + 15min + rt_lag_minutes`.
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "series_id",
                "source",
                "dataset_id",
                "timestamp_utc",
                "value",
                "unit",
                "first_usable_origin_utc",
            ]
        )

    df = raw_df.copy()
    df["start_utc"] = pd.to_datetime(df["startTime"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["start_utc", "value"]).sort_values("start_utc")

    df["timestamp_utc"] = df["start_utc"].dt.floor("15min")
    grouped = df.groupby("timestamp_utc", as_index=False)["value"].mean()
    grouped["series_id"] = spec.series_id
    grouped["source"] = "fingrid_extra"
    grouped["dataset_id"] = str(spec.dataset_id)
    grouped["unit"] = spec.unit
    grouped["first_usable_origin_utc"] = compute_rt_first_usable(
        grouped["timestamp_utc"], lag_minutes=rt_lag_minutes, period_minutes=15
    )
    return grouped[
        [
            "series_id",
            "source",
            "dataset_id",
            "timestamp_utc",
            "value",
            "unit",
            "first_usable_origin_utc",
        ]
    ]


def fetch_fingrid_dataset_chunked(
    dataset_id: int,
    start_utc: datetime,
    end_utc: datetime,
    api_key: str,
    min_interval_s: float = 2.1,
) -> pd.DataFrame:
    """Fetch a Fingrid dataset over [start_utc, end_utc] with rate limiting."""
    import requests

    session = requests.Session()
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    rows: list[dict] = []
    cur = start_utc
    step = timedelta(days=35)

    while cur < end_utc:
        nxt = min(end_utc, cur + step)
        page = 1
        while True:
            params = {
                "startTime": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": nxt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "format": "json",
                "pageSize": 20000,
                "page": page,
            }
            for attempt in range(4):
                resp = session.get(
                    FINGRID_API_URL.format(dataset_id=dataset_id),
                    params=params,
                    headers=headers,
                    timeout=60,
                )
                time.sleep(min_interval_s)
                if resp.status_code == 429:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                break
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", []) if isinstance(payload, dict) else payload
            if not data:
                break
            rows.extend(data)
            pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
            last_page = pagination.get("lastPage", 1)
            if page >= last_page:
                break
            page += 1
        cur = nxt

    return pd.DataFrame(rows)


def load_extra_inputs(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame | None:
    """Load `data/extra_inputs.parquet` (or `.csv`) if present on disk."""
    data_dir = Path(data_dir)
    for name in ("extra_inputs.parquet", "extra_inputs.csv"):
        p = data_dir / name
        if p.exists():
            return read_long(p)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch optional extra Fingrid series (pre-June aFRR/mFRR & 181/188/191/177 RT).")
    parser.add_argument("--pre-june-start", default="2026-04-01T00:00:00Z", help="Start UTC for pre-June aFRR/mFRR lag buffer.")
    parser.add_argument("--rt-start", default="2026-05-31T21:00:00Z", help="Start UTC for 181/188/191/177 real-time series.")
    parser.add_argument("--end", default="2026-11-02T07:00:00Z", help="End UTC.")
    parser.add_argument("--rt-lag-minutes", type=int, default=30, help="Availability lag in minutes after quarter end (default: 30, or 10 if jury confirms).")
    parser.add_argument("--out", default=str(DEFAULT_DATA_DIR / "extra_inputs.parquet"), help="Output parquet path.")
    args = parser.parse_args()

    api_key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not api_key:
        for env_candidate in (Path(".env"), Path("keys.env")):
            if env_candidate.is_file():
                for line in env_candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("FINGRID_API_KEY") and "=" in line:
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if api_key:
                break
    if not api_key:
        print("INFO: FINGRID_API_KEY is not set. Skipping live API fetch (pipeline uses built-in implied aFRR/mFRR volume-pricing features).")
        return

    existing_path = Path(args.out)
    existing_rt: pd.DataFrame | None = None
    if existing_path.exists():
        old_df = pd.read_parquet(existing_path)
        rt_ids = {s.series_id for s in REALTIME_FINGRID_SERIES}
        existing_rt = old_df.loc[old_df["series_id"].isin(rt_ids)].copy()

    frames: list[pd.DataFrame] = []
    pre_start = pd.to_datetime(args.pre_june_start, utc=True).to_pydatetime()
    pre_end = pd.to_datetime("2026-05-31T21:00:00Z", utc=True).to_pydatetime()
    rt_start = pd.to_datetime(args.rt_start, utc=True).to_pydatetime()
    rt_end = pd.to_datetime(args.end, utc=True).to_pydatetime()

    for spec in HISTORICAL_REGULATION_SERIES:
        print(f"Fetching pre-June {spec.series_id} (dataset {spec.dataset_id})...")
        raw = fetch_fingrid_dataset_chunked(spec.dataset_id, pre_start, pre_end, api_key)
        if not raw.empty:
            frames.append(resample_3min_to_quarter_hour(raw, spec, rt_lag_minutes=spec.default_lag_minutes))

    if existing_rt is not None and len(existing_rt) > 50000 and args.rt_lag_minutes == 30:
        print(f"Reusing {len(existing_rt):,} cached 3-min real-time rows from {existing_path}...")
        frames.append(existing_rt)
    else:
        for spec in REALTIME_FINGRID_SERIES:
            print(f"Fetching real-time {spec.series_id} (dataset {spec.dataset_id}) with lag={args.rt_lag_minutes}m...")
            raw = fetch_fingrid_dataset_chunked(spec.dataset_id, rt_start, rt_end, api_key)
            if not raw.empty:
                frames.append(resample_3min_to_quarter_hour(raw, spec, rt_lag_minutes=args.rt_lag_minutes))

    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        print(f"Saved {len(out_df):,} extra rows to {out_path}")


if __name__ == "__main__":
    main()
