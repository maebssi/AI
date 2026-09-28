"""AI허브 '제조현장 이송장치의 열화 예지보전 멀티모달 데이터' 전처리.

원천(TS/VS zip: 센서 csv + 열화상 bin) 과 라벨(TL/VL zip: json) 을 세션 단위로 짝지어
  - frames.parquet : 프레임(1초) 단위 센서·외부환경·열화상 통계·상태 라벨
  - thermal_{split}.npy : 열화상 30x40 썸네일 (float16, frames 행 순서와 동일)
을 만든다.

실행: python -m maebssi.data.build_dataset [--workers 8] [--limit N]
"""
import argparse
import io
import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from maebssi.config import EXTERNAL, PROCESSED_DIR, RAW_DIR, SENSORS, THUMB_SHAPE

SPLITS = {"train": ("Training", "TS", "TL"), "valid": ("Validation", "VS", "VL")}
SESSION_RE = re.compile(r"^[TV][SL]_(agv|oht)_(\d+)_(\w+?)_(\d{4})_(\d{4})\.zip$")


def list_sessions(split: str) -> list[tuple[Path, Path]]:
    folder, src_prefix, lbl_prefix = SPLITS[split]
    src_dir = RAW_DIR / folder / "01.원천데이터"
    lbl_dir = RAW_DIR / folder / "02.라벨링데이터"
    pairs = []
    for lbl in sorted(lbl_dir.glob(f"{lbl_prefix}_*.zip")):
        src = src_dir / lbl.name.replace(lbl_prefix, src_prefix, 1)
        if src.exists():
            pairs.append((src, lbl))
    return pairs


def thermal_stats(img: np.ndarray) -> dict:
    flat = img.ravel()
    mean, std = float(flat.mean()), float(flat.std())
    p50, p90, p99 = np.percentile(flat, [50, 90, 99])
    h, w = img.shape
    center = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return {
        "ir_mean": mean,
        "ir_std": std,
        "ir_min": float(flat.min()),
        "ir_max": float(flat.max()),
        "ir_p50": float(p50),
        "ir_p90": float(p90),
        "ir_p99": float(p99),
        "ir_hot_frac": float((flat > mean + 2 * std).mean()),
        "ir_center_mean": float(center.mean()),
        "ir_grad": float(np.abs(np.diff(img, axis=0)).mean() + np.abs(np.diff(img, axis=1)).mean()),
    }


def thumbnail(img: np.ndarray) -> np.ndarray:
    th, tw = THUMB_SHAPE
    h, w = img.shape
    return img.reshape(th, h // th, tw, w // tw).mean(axis=(1, 3)).astype(np.float16)


def _first(d: dict, key: str):
    v = d.get(key)
    return v[0] if isinstance(v, list) and v else {}


def process_session(src: Path, lbl: Path, split: str):
    m = SESSION_RE.match(lbl.name)
    kind, _, device, mmdd, hhmm = m.groups()
    session_id = f"{device}_{mmdd}_{hhmm}"
    rows, thumbs = [], []
    with zipfile.ZipFile(src) as zs, zipfile.ZipFile(lbl) as zl:
        bins = {Path(n).stem: n for n in zs.namelist() if n.endswith(".bin")}
        for name in sorted(zl.namelist()):
            if not name.endswith(".json"):
                continue
            stem = Path(name).stem
            if stem not in bins:
                continue
            j = json.loads(zl.read(name))
            meta = j["meta_info"][0]
            sensor = j["sensor_data"][0]
            ext = j["external_data"][0]
            ir = _first(j["ir_data"][0], "temp_max")
            state = int(_first(j["annotations"][0], "tagging")["state"])
            img = np.load(io.BytesIO(zs.read(bins[stem])))

            row = {
                "split": split,
                "kind": kind,
                "device_id": meta["device_id"],
                "session_id": session_id,
                "frame_id": stem,
                "date": meta["collection_date"],
                "time": meta["collection_time"],
                "cumulative_operating_day": float(meta.get("cumulative_operating_day") or 0),
                "equipment_history": float(meta.get("equipment_history") or 0),
                "manufacturer": meta.get("device_manufacturer", ""),
                "state": state,
            }
            for s in SENSORS:
                row[s] = float(_first(sensor, s).get("value", np.nan))
            for e in EXTERNAL:
                row[e] = float(_first(ext, e).get("value", np.nan))
            row["ir_label_tmax"] = float(ir.get("value_TGmx", np.nan))
            row["ir_label_x"] = float(ir.get("X_Tmax", np.nan))
            row["ir_label_y"] = float(ir.get("Y_Tmax", np.nan))
            row.update(thermal_stats(img))
            rows.append(row)
            thumbs.append(thumbnail(img))
    df = pd.DataFrame(rows)
    return df, (np.stack(thumbs) if thumbs else np.zeros((0, *THUMB_SHAPE), np.float16))


def build(split: str, workers: int, limit: int | None):
    pairs = list_sessions(split)
    if limit:
        pairs = pairs[:limit]
    print(f"[{split}] sessions: {len(pairs)}")
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_session, s, l, split): i for i, (s, l) in enumerate(pairs)}
        for n, f in enumerate(as_completed(futs), 1):
            results[futs[f]] = f.result()
            if n % 20 == 0 or n == len(pairs):
                print(f"  {n}/{len(pairs)}", flush=True)
    dfs = [results[i][0] for i in range(len(pairs))]
    ths = [results[i][1] for i in range(len(pairs))]
    df = pd.concat(dfs, ignore_index=True)
    thumbs = np.concatenate(ths)
    df["ts"] = pd.to_datetime("2024-" + df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S")
    return df, thumbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="split 당 세션 수 제한 (디버그)")
    args = ap.parse_args()
    all_df = []
    for split in SPLITS:
        df, thumbs = build(split, args.workers, args.limit)
        np.save(PROCESSED_DIR / f"thermal_{split}.npy", thumbs)
        df["thumb_idx"] = np.arange(len(df))
        all_df.append(df)
        print(f"[{split}] frames={len(df)} state dist={df['state'].value_counts().sort_index().to_dict()}")
    out = pd.concat(all_df, ignore_index=True)
    out.to_parquet(PROCESSED_DIR / "frames.parquet", index=False)
    print("saved", PROCESSED_DIR / "frames.parquet", out.shape)


if __name__ == "__main__":
    main()
