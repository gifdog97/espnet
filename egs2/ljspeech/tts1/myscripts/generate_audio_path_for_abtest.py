import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ====== Design (AB test) ======
WORKERS = 120
SAMPLES_PER_PAIR = 20  # "pair samples" per (settingA, settingB)
SAMPLES_INCLUDING_SWAPPED = (
    2 * SAMPLES_PER_PAIR
)  # Counting swapped presentation as separate samples -> 40 per pair
WORKERS_PER_SAMPLE = 6  # ratings per pair sample

SETTINGS_NK: List[Tuple[int, int]] = [
    (20, 256),
    (40, 256),
    (80, 4096),
    (120, 4096),
]
PAIR_KEYS: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list(
    itertools.combinations(SETTINGS_NK, 2)
)

MODEL_NAME = "tacotron2"
CONTINUATION_CSV = Path("csv/continuation_result_10s.csv")

# Your original template (temperature is inserted)
WAV_DIR_TEMPLATE = "/work/gk77/k77035/espnet/egs2/ljspeech/tts1/myscripts/audio_cut_10s/{N}-{K}-{temperature}"


# ====== CSV: extract (N,K)->temperature ======
def extract_temperature_map(
    result_csv: Path,
    model_name: str,
    target_nk: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """
    Reads csv/continuation_result.csv and returns:
      {(N,K): temperature}
    with checks:
      - only model_name rows (setting startswith model_name)
      - temperature must be present
      - for each target (N,K), temperature is unique
    """
    target_set = set(target_nk)
    seen: Dict[Tuple[int, int], set[float]] = {nk: set() for nk in target_set}

    with result_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            setting = row.get("setting", "")
            if not setting.startswith(model_name):
                continue
            temp_str = row.get("temperature", "")
            if temp_str == "":
                continue

            # setting format: tacotron2-N-K
            try:
                _model, N_str, K_str = setting.split("-")
                N, K = int(N_str), int(K_str)
            except Exception as e:
                raise ValueError(f"Unexpected setting format: {setting}") from e

            nk = (N, K)
            if nk not in target_set:
                continue

            try:
                t = float(temp_str)
            except Exception as e:
                raise ValueError(
                    f"Bad temperature value: {temp_str} (setting={setting})"
                ) from e

            seen[nk].add(t)

    temp_map: Dict[Tuple[int, int], float] = {}
    missing = []
    non_unique = []
    for nk in target_nk:
        temps = seen.get(nk, set())
        if len(temps) == 0:
            missing.append(nk)
        elif len(temps) > 1:
            non_unique.append((nk, sorted(temps)))
        else:
            temp_map[nk] = next(iter(temps))

    if missing:
        raise ValueError(f"Missing (N,K) in {result_csv}: {missing}")
    if non_unique:
        raise ValueError(f"Non-unique temperature for some (N,K): {non_unique}")

    return temp_map


# ====== Data collection ======
def collect_wavs_for_setting(
    wav_dir: Path,
) -> Dict[str, str]:
    """
    Returns: {stem -> audio_path}
    - Keeps only *.wav whose stem ends with `require_suffix` (to match your prior filtering).
    - Pairing is by identical stem.
    """
    out: Dict[str, str] = {}
    for p in sorted(wav_dir.glob("*.wav")):
        stem = p.stem
        if stem in out:
            raise ValueError(f"Duplicate stem in {wav_dir}: {stem}")
        out[stem] = str(p)
    return out


def build_setting_dirs_from_csv(
    result_csv: Path,
    model_name: str,
    target_nk: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Path]:
    """
    Use continuation_result.csv to determine unique temperature for each (N,K),
    then build wav_dir via WAV_DIR_TEMPLATE.
    """
    temp_map = extract_temperature_map(result_csv, model_name, target_nk)

    dirs: Dict[Tuple[int, int], Path] = {}
    for N, K in target_nk:
        t = temp_map[(N, K)]
        wav_dir = Path(WAV_DIR_TEMPLATE.format(N=N, K=K, temperature=t))
        if not wav_dir.exists():
            raise FileNotFoundError(
                f"wav_dir not found for (N,K)=({N},{K}), temperature={t}: {wav_dir}"
            )
        dirs[(N, K)] = wav_dir
    return dirs


@dataclass(frozen=True)
class PairItem:
    pair_id: int
    sample_id: int
    stem: str

    N_a: int
    K_a: int
    N_b: int
    K_b: int

    audio_a: str
    audio_b: str


def build_pair_item_sets(
    setting_to_wavs: Dict[Tuple[int, int], Dict[str, str]],
    base_seed: int,
) -> List[List[PairItem]]:
    # all_item_sets[sid] = list of PairItem for sample_id=sid, across all pairs (including swapped)
    # i.e., all_item_sets[0] has 6x2=12 items , all_item_sets[1] has 12 items, ..., all_item_sets[19] has 12 items
    # e.g., all_item_sets[0] has id=0 samples (and their swaps) for 6 pairs.
    all_item_sets: List[List[PairItem]] = [[] for _ in range(SAMPLES_PER_PAIR)]
    for pair_id, (a, b) in enumerate(PAIR_KEYS):
        wavs_a = setting_to_wavs[a]
        wavs_b = setting_to_wavs[b]
        common = sorted(set(wavs_a.keys()) & set(wavs_b.keys()))
        if len(common) < SAMPLES_PER_PAIR:
            raise ValueError(
                f"Not enough common stems for pair {a} vs {b}: "
                f"common={len(common)} < {SAMPLES_PER_PAIR}"
            )
        while True:
            rng = np.random.default_rng((base_seed * 1000 + pair_id) % (2**32))
            common_arr = np.array(common, dtype=object)
            rng.shuffle(common_arr)
            chosen = common_arr[:SAMPLES_PER_PAIR].tolist()
            if all(
                stem not in [item.stem for item in all_item_sets[sid]]
                for sid, stem in enumerate(chosen)
            ):
                break
            print("Overlap found, reshuffling...")
            base_seed += 1  # try a different shuffle if there's overlap
        (N_a, K_a), (N_b, K_b) = a, b
        for sid, stem in enumerate(chosen):
            all_item_sets[sid].append(
                PairItem(
                    pair_id=pair_id,
                    sample_id=sid,
                    stem=stem,
                    N_a=N_a,
                    K_a=K_a,
                    N_b=N_b,
                    K_b=K_b,
                    audio_a=wavs_a[stem],
                    audio_b=wavs_b[stem],
                )
            )
            # include swapped presentation as separate samples
            all_item_sets[sid].append(
                PairItem(
                    pair_id=pair_id,
                    sample_id=sid
                    + SAMPLES_PER_PAIR,  # separate sample_id for swapped presentation
                    stem=stem,
                    N_a=N_b,
                    K_a=K_b,
                    N_b=N_a,
                    K_b=K_a,
                    audio_a=wavs_b[stem],
                    audio_b=wavs_a[stem],
                )
            )
    return all_item_sets


# ====== Assignment ======
def assign_items_to_workers(
    items: List[List[PairItem]],
    n_workers: int,
    workers_per_sample: int,
    base_seed: int,
) -> Dict[int, List[PairItem]]:
    """
    Perfectly balanced construction:
      - 120 workers, 6 workers per sample => 20 disjoint worker-groups
      - total items = 6 pairs * 40 = 240
      - each group rates 240/20=12 items => each worker rates 12 items
      - each item gets exactly 6 workers
    """
    if n_workers != WORKERS:
        raise ValueError(f"Expected WORKERS={WORKERS}, got n_workers={n_workers}")
    if workers_per_sample != WORKERS_PER_SAMPLE:
        raise ValueError(
            f"Expected WORKERS_PER_SAMPLE={WORKERS_PER_SAMPLE}, got {workers_per_sample}"
        )
    if sum(len(x) for x in items) != len(PAIR_KEYS) * SAMPLES_INCLUDING_SWAPPED:
        raise ValueError(
            f"Expected {len(PAIR_KEYS) * SAMPLES_INCLUDING_SWAPPED} total items, got {sum(len(x) for x in items)}"
        )
    if n_workers % workers_per_sample != 0:
        raise ValueError("n_workers must be divisible by workers_per_sample")

    n_groups = n_workers // workers_per_sample
    if len(items) % n_groups != 0:
        raise ValueError("Total items must be divisible by number of worker-groups")

    rng = np.random.default_rng(base_seed)

    worker_ids = np.arange(n_workers, dtype=int)
    rng.shuffle(worker_ids)

    groups: List[List[int]] = [
        worker_ids[i * workers_per_sample : (i + 1) * workers_per_sample].tolist()
        for i in range(n_groups)
    ]

    worker_to_items: Dict[int, List[PairItem]] = dict()

    for idx, it in enumerate(items):
        g = groups[idx % n_groups]
        for wid in g:
            worker_to_items[wid] = it

    return worker_to_items


def constrained_shuffle_pairs(
    items: List[PairItem],
    rng: np.random.Generator,
    max_passes: int = 2000,
) -> List[PairItem]:
    """Avoid consecutive same pair_id / same stem when possible."""
    if len(items) <= 2:
        return items

    arr = items.copy()
    rng.shuffle(arr)

    def bad(i: int) -> bool:
        if i <= 0:
            return False
        a, b = arr[i - 1], arr[i]
        if a.pair_id == b.pair_id:
            return True
        if a.stem == b.stem:
            return True
        return False

    for _ in range(max_passes):
        idxs = [i for i in range(1, len(arr)) if bad(i)]
        if not idxs:
            break
        i = idxs[0]
        found = False
        for j in range(i + 1, len(arr)):
            arr[i], arr[j] = arr[j], arr[i]
            ok = True
            for k in (i, i + 1, j, j + 1):
                if 1 <= k < len(arr) and bad(k):
                    ok = False
                    break
            if ok:
                found = True
                break
            arr[i], arr[j] = arr[j], arr[i]
        if not found:
            tail = arr[i:]
            rng.shuffle(tail)
            arr[i:] = tail

    return arr


# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=Path, required=True, help="Output assignment CSV")
    ap.add_argument("--base_seed", type=int, default=20260219)
    ap.add_argument("--n_workers", type=int, default=WORKERS)
    ap.add_argument("--continuation_csv", type=Path, default=CONTINUATION_CSV)
    ap.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = ap.parse_args()

    # 1) Determine wav dirs using CSV-derived temperatures
    setting_dirs = build_setting_dirs_from_csv(
        result_csv=args.continuation_csv,
        model_name=args.model_name,
        target_nk=SETTINGS_NK,
    )

    # 2) Load wavs per setting
    setting_to_wavs: Dict[Tuple[int, int], Dict[str, str]] = {}
    for nk, wav_dir in setting_dirs.items():
        setting_to_wavs[nk] = collect_wavs_for_setting(wav_dir)

    # 3) Build AB pair items (10 pairs * 20 samples * 2 with swapped presentation = 400 items)
    items = build_pair_item_sets(setting_to_wavs, base_seed=args.base_seed)

    # 4) Assign items to workers (each worker 20, each item 6 workers)
    worker_to_items = assign_items_to_workers(
        items=items,
        n_workers=args.n_workers,
        workers_per_sample=WORKERS_PER_SAMPLE,
        base_seed=args.base_seed,
    )

    # 5) Emit rows
    rows = []
    for wid in range(args.n_workers):
        its = worker_to_items[wid]

        wseed = (args.base_seed * 1000003 + wid) % (2**32)
        wrng = np.random.default_rng(wseed)
        ordered = constrained_shuffle_pairs(its, wrng)

        for order_idx, it in enumerate(ordered):
            rows.append(
                {
                    "worker_id": wid,
                    "order": order_idx,
                    "pair_id": it.pair_id,
                    "sample_id": it.sample_id,
                    "stem": it.stem,
                    "A_N": it.N_a,
                    "A_K": it.K_a,
                    "B_N": it.N_b,
                    "B_K": it.K_b,
                    "A_audio_path": it.audio_a,
                    "B_audio_path": it.audio_b,
                }
            )

    out_df = (
        pd.DataFrame(rows).sort_values(["worker_id", "order"]).reset_index(drop=True)
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}  (rows={len(out_df)})")

    # ====== Sanity checks ======
    # 1) each worker gets 40 items
    exp_per_worker = (sum(len(x) for x in items) * WORKERS_PER_SAMPLE) // args.n_workers
    c1 = out_df.groupby("worker_id").size()
    if not (c1 == exp_per_worker).all():
        bad = c1[c1 != exp_per_worker].head(20)
        raise RuntimeError(
            f"Not all workers have {exp_per_worker} items. Examples:\n{bad}"
        )

    # 2) each (pair_id,sample_id) gets exactly 6 workers
    c2 = out_df.groupby(["pair_id", "sample_id"]).size()
    if not (c2 == WORKERS_PER_SAMPLE).all():
        bad = c2[c2 != WORKERS_PER_SAMPLE].head(20)
        raise RuntimeError(
            f"Not all (pair_id,sample_id) have {WORKERS_PER_SAMPLE} ratings. Examples:\n{bad}"
        )

    # 3) stem consistency
    def stem_of(p: str) -> str:
        return Path(p).stem

    mismatch = out_df[
        (out_df["stem"] != out_df["A_audio_path"].map(stem_of))
        | (out_df["stem"] != out_df["B_audio_path"].map(stem_of))
    ]
    if len(mismatch) > 0:
        raise RuntimeError(
            f"Stem mismatch found in {len(mismatch)} rows (pairing bug)."
        )

    # 4) Each worker is assigned 10 combinations of settings and its swapped version, different stems
    def assigned_properly(group: pd.DataFrame) -> bool:
        pairs = set()
        stems = set()
        for _, row in group.iterrows():
            pair = (
                (int(row["A_N"]), int(row["A_K"])),
                (int(row["B_N"]), int(row["B_K"])),
            )
            pairs.add(pair)
            stems.add(row["stem"])
        pairs_including_swap = set(PAIR_KEYS) | set(((b, a) for (a, b) in PAIR_KEYS))
        return pairs_including_swap == pairs and len(stems) == len(PAIR_KEYS)

    bad_workers = out_df.groupby("worker_id").apply(assigned_properly)
    if not bad_workers.all():
        bad = bad_workers[bad_workers == False].head(20)
        raise RuntimeError(f"Workers with incorrect assignments: {list(bad.index)}")

    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
