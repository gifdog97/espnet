#!/usr/bin/env python3


import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WORKERS = 250
SAMPLES_PER_SETTING = 50
WORKER_PER_SAMPLE = 10


def extract_settings(
    result_csv: str, model_name: str = "tacotron2"
) -> List[Tuple[str, str, str, float]]:
    settings = []
    with open(result_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["setting"].startswith(model_name):
                continue
            if row["temperature"] == "":
                continue
            setting, temperature = row["setting"], float(row["temperature"])
            model_name, N, K = setting.split("-")
            settings.append((model_name, N, K, temperature))
    return settings


def prepare_df():
    settings = extract_settings("csv/continuation_result.csv")
    assert len(settings) == 21, f"Expected 21 settings, got {len(settings)}"
    path_template_t = "/work/gk77/k77035/espnet/egs2/ljspeech/tts1/exp/fixed_{N}-{K}_dedup/tts_train_raw_phn_none/continuation_{temperature}/dev/wav"
    audio_dirs = [
        Path(path_template_t.format(N=N, K=K, temperature=temperature))
        for _, N, K, temperature in settings
    ]
    setting_ids = [i for i in range(len(settings)) for _ in range(SAMPLES_PER_SETTING)]
    sample_ids = [i for i in range(SAMPLES_PER_SETTING)] * len(settings)
    audio_paths = []
    for wav_dir in audio_dirs:
        this_audio_paths = []
        for audio_path in sorted(wav_dir.glob("*.wav")):
            if not audio_path.stem.endswith("-0"):
                continue
            this_audio_paths.append(str(audio_path))
            if len(this_audio_paths) == SAMPLES_PER_SETTING:
                break
        audio_paths.extend(this_audio_paths)
    df = pd.DataFrame(
        {
            "setting_id": setting_ids,
            "sample_id": sample_ids,
            "audio_path": audio_paths,
        }
    )

    return df


@dataclass(frozen=True)
class Item:
    setting_id: int
    sample_id: int
    audio_path: str
    speaker_id: Optional[str] = None
    text_id: Optional[str] = None


def build_items(df: pd.DataFrame) -> Dict[int, List[Item]]:
    required = {"setting_id", "sample_id", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    has_spk = "speaker_id" in df.columns
    has_txt = "text_id" in df.columns

    items_by_setting: Dict[int, List[Item]] = {}
    for _, r in df.iterrows():
        it = Item(
            setting_id=int(r["setting_id"]),
            sample_id=int(r["sample_id"]),
            audio_path=str(r["audio_path"]),
            speaker_id=str(r["speaker_id"])
            if has_spk and pd.notna(r["speaker_id"])
            else None,
            text_id=str(r["text_id"]) if has_txt and pd.notna(r["text_id"]) else None,
        )
        items_by_setting.setdefault(it.setting_id, []).append(it)

    # sort by sample_id for stability
    for sid in items_by_setting:
        items_by_setting[sid] = sorted(items_by_setting[sid], key=lambda x: x.sample_id)

    return items_by_setting


def assign_one_setting(
    items: List[Item],
    worker_ids: np.ndarray,
    rng: np.random.Generator,
) -> Dict[Tuple[int, int], List[int]]:
    """
    For a single setting with 50 samples:
      - 250 workers
      - each worker gets exactly 2 samples per setting (=> 42 total)
      - each sample gets exactly 10 workers
      - no worker is assigned the same sample twice

    Construction:
      1) Shuffle workers.
      2) Split into 50 groups of 5 (round 1 assignment).
      3) Round 2 uses a cyclic shift of groups by +1 item index.
         => each worker gets two different samples, each sample gets 5+5=10 workers.
    """
    n_items = len(items)
    if n_items != SAMPLES_PER_SETTING:
        raise ValueError(
            f"Expected {SAMPLES_PER_SETTING} items per setting, got {n_items}"
        )

    n_workers = len(worker_ids)
    if n_workers != WORKERS:
        raise ValueError(f"Expected {WORKERS} workers, got {n_workers}")

    # Shuffle workers and split into 50 groups of 5
    w = worker_ids.copy()
    rng.shuffle(w)
    groups = [w[i * 5 : (i + 1) * 5].tolist() for i in range(50)]

    # For item j: assign groups[j] (round1) + groups[j-1] (round2 shifted by +1)
    # This guarantees disjointness within an item because groups are disjoint.
    assignment: Dict[Tuple[int, int], List[int]] = {}
    for j, it in enumerate(items):
        workers_for_item = groups[j] + groups[(j - 1) % 50]
        if len(workers_for_item) != WORKER_PER_SAMPLE:
            raise RuntimeError(
                f"Internal error: not {WORKER_PER_SAMPLE} workers per item"
            )
        assignment[(it.setting_id, it.sample_id)] = workers_for_item

    return assignment


def flatten_to_worker_lists(
    items_by_setting: Dict[int, List[Item]],
    per_setting_assignment: Dict[Tuple[int, int], List[int]],
    n_workers: int,
) -> Dict[int, List[Item]]:
    worker_to_items: Dict[int, List[Item]] = {wid: [] for wid in range(n_workers)}
    # build lookup for Item
    lookup: Dict[Tuple[int, int], Item] = {}
    for sid, items in items_by_setting.items():
        for it in items:
            lookup[(it.setting_id, it.sample_id)] = it

    for (sid, sample_id), wids in per_setting_assignment.items():
        it = lookup[(sid, sample_id)]
        for wid in wids:
            worker_to_items[wid].append(it)

    return worker_to_items


def constrained_shuffle(
    items: List[Item],
    rng: np.random.Generator,
    max_passes: int = 2000,
) -> List[Item]:
    """
    Try to reduce consecutive same speaker_id / text_id.
    It's a lightweight heuristic: shuffle then fix by swaps.
    If speaker_id/text_id is None, ignores that constraint.
    """
    if len(items) <= 2:
        return items

    arr = items.copy()
    rng.shuffle(arr)

    def bad(i: int) -> bool:
        if i <= 0:
            return False
        a, b = arr[i - 1], arr[i]
        if (
            a.speaker_id is not None
            and b.speaker_id is not None
            and a.speaker_id == b.speaker_id
        ):
            return True
        if a.text_id is not None and b.text_id is not None and a.text_id == b.text_id:
            return True
        return False

    # Greedy swap repair
    for _ in range(max_passes):
        idxs = [i for i in range(1, len(arr)) if bad(i)]
        if not idxs:
            break
        i = idxs[0]
        # find a j to swap with that improves
        found = False
        for j in range(i + 1, len(arr)):
            # swap temporarily
            arr[i], arr[j] = arr[j], arr[i]
            ok = True
            for k in (i, i + 1, j, j + 1):
                if 1 <= k < len(arr) and bad(k):
                    ok = False
                    break
            if ok:
                found = True
                break
            # revert
            arr[i], arr[j] = arr[j], arr[i]
        if not found:
            # give up on this violation; reshuffle tail a bit
            tail = arr[i:]
            rng.shuffle(tail)
            arr[i:] = tail

    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=Path, required=True, help="Output assignment CSV")
    ap.add_argument("--n_workers", type=int, default=250)
    ap.add_argument("--base_seed", type=int, default=20260214)
    args = ap.parse_args()

    df = prepare_df()
    items_by_setting = build_items(df)

    # sanity checks
    setting_ids = sorted(items_by_setting.keys())
    if len(setting_ids) != 21:
        raise ValueError(f"Expected 21 settings, got {len(setting_ids)}: {setting_ids}")

    for sid, items in items_by_setting.items():
        if len(items) != 50:
            raise ValueError(f"Setting {sid}: expected 50 samples, got {len(items)}")
        sample_ids = [it.sample_id for it in items]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Setting {sid}: sample_id has duplicates")

    worker_ids = np.arange(args.n_workers, dtype=int)
    rng = np.random.default_rng(args.base_seed)

    # Build per-setting assignments
    per_setting_assignment: Dict[Tuple[int, int], List[int]] = {}
    for sid in setting_ids:
        # use derived seed per setting for reproducibility
        local_rng = np.random.default_rng((args.base_seed * 1000 + sid) % (2**32))
        a = assign_one_setting(items_by_setting[sid], worker_ids, local_rng)
        per_setting_assignment.update(a)

    # Convert to worker -> list of items (should be 42 each)
    worker_to_items = flatten_to_worker_lists(
        items_by_setting, per_setting_assignment, args.n_workers
    )

    # Build final rows with per-worker ordering
    rows = []
    for wid in range(args.n_workers):
        its = worker_to_items[wid]
        if len(its) != 42:
            raise RuntimeError(f"Worker {wid}: expected 42 items, got {len(its)}")

        # worker-specific seed so you can reproduce exact order
        wseed = (args.base_seed * 1000003 + wid) % (2**32)
        wrng = np.random.default_rng(wseed)

        ordered = constrained_shuffle(its, wrng)

        for order_idx, it in enumerate(ordered):
            rows.append(
                {
                    "worker_id": wid,
                    "order": order_idx,
                    "setting_id": it.setting_id,
                    "sample_id": it.sample_id,
                    "audio_path": it.audio_path,
                    "speaker_id": it.speaker_id,
                    "text_id": it.text_id,
                    "worker_seed": int(wseed),
                }
            )

    out_df = (
        pd.DataFrame(rows).sort_values(["worker_id", "order"]).reset_index(drop=True)
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}  (rows={len(out_df)})")

    # quick diagnostics
    # 1) each worker gets 2 per setting
    check = out_df.groupby(["worker_id", "setting_id"]).size()
    if not (check == 2).all():
        bad = check[check != 2].head(20)
        raise RuntimeError(f"Not all (worker,setting) have 2 items. Examples:\n{bad}")

    # 2) each (setting,sample) gets 10 workers
    check2 = out_df.groupby(["setting_id", "sample_id"]).size()
    if not (check2 == 10).all():
        bad = check2[check2 != 10].head(20)
        raise RuntimeError(
            f"Not all (setting,sample) have 10 ratings. Examples:\n{bad}"
        )

    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
