"""
Generic phase runner — runs a list of variants given as args.

Usage:
    python phase_runner.py <phase_id> <variant_id> [variant_id ...]

Example:
    python phase_runner.py phase2 V8 V9 V10 V11

Behavior:
- Runs each variant sequentially via run_variant.py subprocess
- Auto-appends to .lab/log.md and .lab/results.tsv
- Generates phase_<phase_id>_summary.{md,json} when done
"""
import sys
import json
import time
import pickle
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parent
LAB_ROOT = LAB_DIR.parent
PYTHON = r"C:\Users\Acer\anaconda3\envs\pcgnn\python.exe"

WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
TILE_COLORS = {
    WALL: [0.15, 0.15, 0.15], FLOOR: [0.95, 0.95, 0.95],
    PLAYER: [0.20, 0.80, 0.20], ENEMY: [0.90, 0.20, 0.20],
}


def log(phase_id, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{phase_id}] {msg}"
    print(line, flush=True)
    with open(LAB_ROOT / f"{phase_id}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_variant(phase_id, vid):
    out_dir = LAB_DIR / f"expv-{vid}-{phase_id}"
    log_path = LAB_DIR / f"expv-{vid}-{phase_id}.log"
    log(phase_id, f">>> Starting {vid}")
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                [PYTHON, str(LAB_DIR / "run_variant.py"), vid, phase_id],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=4500,
            )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        log(phase_id, f"<<< {vid} {'OK' if ok else 'FAILED rc=' + str(proc.returncode)} in {elapsed:.0f}s")
        return ok
    except subprocess.TimeoutExpired:
        log(phase_id, f"<<< {vid} TIMEOUT")
        return False
    except Exception as e:
        log(phase_id, f"<<< {vid} EXCEPTION: {e}")
        return False


def append_log_md(phase_id, vid, result, exp_num, parent_exp):
    if not result or result.get("status") != "OK":
        block = (
            f"\n## Experiment {exp_num} — {phase_id}/{vid} (FAILED)\n"
            f"Status: {result.get('status', 'UNKNOWN') if result else 'NO_RESULT'}\n"
        )
    else:
        h = result.get("turns_histogram", [0]*10)
        hist_str = ", ".join(f"{l}:{c}" for l, c in zip(
            ["0","1","2","3","4","5","6","7","≥8","unsolv"], h))
        block = (
            f"\n## Experiment {exp_num} — {phase_id}/{vid}\n"
            f"Description: {result.get('description', '')}\n"
            f"Result: non_l={result['non_l_pattern_rate']:.4f}, "
            f"solv={result['solvability']:.4f}, "
            f"diag={result['diagonal_stripes_count']}/{result['n_maps']}, "
            f"empty={result['near_empty_count']}/{result['n_maps']}, "
            f"full={result['near_full_count']}/{result['n_maps']}, "
            f"div={result['astar_div_pairs_mean']:.4f}, "
            f"diff={result['astar_diff_mean']:.4f}\n"
            f"Turns: {hist_str}\n"
            f"Duration: {result.get('train_seconds', 0):.1f}s\n"
            f"Status: keep ({phase_id})\n"
        )
    with open(LAB_ROOT / "log.md", "a", encoding="utf-8") as f:
        f.write(block)


def append_results_tsv(phase_id, vid, result, exp_num, parent_exp):
    if not result or result.get("status") != "OK":
        row = f"{exp_num}\tresearch/bc-vector-l-pattern\t{parent_exp}\t-\t-1\t-\tcrash\t-\t{phase_id}/{vid}: FAILED\n"
    else:
        n = result["n_maps"]
        compound = (
            result["non_l_pattern_rate"]
            * result["solvability"]
            * (1 - result["diagonal_stripes_count"] / n)
            * (1 - (result["near_empty_count"] + result["near_full_count"]) / n)
        )
        secondary = (
            f"solv={result['solvability']:.2f} "
            f"diag={result['diagonal_stripes_count']} "
            f"empty={result['near_empty_count']} "
            f"full={result['near_full_count']} "
            f"compound={compound:.3f}"
        )
        desc = f"{phase_id}/{vid}: {result.get('description', '')[:80]}"
        row = (
            f"{exp_num}\tresearch/bc-vector-l-pattern\t{parent_exp}\t-\t"
            f"{result['non_l_pattern_rate']:.6f}\t{secondary}\tkeep\t"
            f"{result.get('train_seconds', 0):.1f}\t{desc}\n"
        )
    with open(LAB_ROOT / "results.tsv", "a", encoding="utf-8") as f:
        f.write(row)


def save_sample_png(phase_id, vid):
    out_dir = LAB_DIR / f"expv-{vid}-{phase_id}"
    maps_path = out_dir / "maps.pkl"
    if not maps_path.exists():
        return
    with open(maps_path, "rb") as f:
        maps = pickle.load(f)
    fig, axs = plt.subplots(1, 10, figsize=(16, 1.8))
    for j in range(min(10, len(maps))):
        lvl = maps[j]
        h, w = lvl.shape
        img = np.zeros((h, w, 3))
        for tile, color in TILE_COLORS.items():
            img[lvl == tile] = color
        axs[j].imshow(img, interpolation="nearest")
        axs[j].axis("off")
    plt.suptitle(f"{phase_id}/{vid}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "sample.png", dpi=120, bbox_inches="tight")
    plt.close()


def compute_compound(r):
    if r.get("status") != "OK":
        return -1.0
    n = r.get("n_maps", 100)
    return (
        r.get("non_l_pattern_rate", 0)
        * r.get("solvability", 0)
        * (1 - r.get("diagonal_stripes_count", 0) / n)
        * (1 - (r.get("near_empty_count", 0) + r.get("near_full_count", 0)) / n)
    )


def write_phase_report(phase_id, variants):
    rows = []
    for vid in variants:
        result_path = LAB_DIR / f"expv-{vid}-{phase_id}" / "result.json"
        if result_path.exists():
            try:
                r = json.loads(result_path.read_text())
                r["status"] = "OK"
                rows.append(r)
            except Exception as e:
                rows.append({"variant_id": vid, "status": f"PARSE_ERROR: {e}"})
        else:
            rows.append({"variant_id": vid, "status": "MISSING"})

    json_path = LAB_ROOT / f"{phase_id}_summary.json"
    md_path = LAB_ROOT / f"{phase_id}_summary.md"
    json_path.write_text(json.dumps(rows, indent=2))

    md = [f"# {phase_id} Summary\n", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    md.append("\n| Rank | Variant | Description | non_l | solv | diag | empty | full | div | diff | Compound |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ranked = sorted(rows, key=lambda r: -compute_compound(r))
    for rank, r in enumerate(ranked, 1):
        if r.get("status") != "OK":
            md.append(f"| - | {r['variant_id']} | — | — | — | — | — | — | — | — | {r['status']} |")
            continue
        md.append(
            f"| {rank} | **{r['variant_id']}** | {r.get('description', '')[:50]} | "
            f"{r['non_l_pattern_rate']:.2f} | {r['solvability']:.2f} | "
            f"{r['diagonal_stripes_count']} | {r['near_empty_count']} | {r['near_full_count']} | "
            f"{r['astar_div_pairs_mean']:.3f} | {r['astar_diff_mean']:.3f} | "
            f"**{compute_compound(r):.3f}** |"
        )
    md_path.write_text("\n".join(md), encoding="utf-8")
    log(phase_id, f"Report: {md_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: phase_runner.py <phase_id> <variant_id> [...]")
        sys.exit(1)
    phase_id = sys.argv[1]
    variants = sys.argv[2:]
    log(phase_id, f"=== {phase_id} START — variants: {variants} ===")
    overall_t0 = time.time()

    # Determine starting exp_num — read existing results.tsv, get max
    tsv = (LAB_ROOT / "results.tsv").read_text().strip().split("\n")
    exp_nums = []
    for line in tsv[1:]:
        parts = line.split("\t")
        if parts and parts[0].isdigit():
            exp_nums.append(int(parts[0]))
    next_exp = (max(exp_nums) + 1) if exp_nums else 0

    # Run each
    for i, vid in enumerate(variants):
        ok = run_variant(phase_id, vid)
        result_path = LAB_DIR / f"expv-{vid}-{phase_id}" / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                result["status"] = "OK"
            except Exception as e:
                result = {"status": f"PARSE_ERROR: {e}"}
        else:
            result = {"status": "NO_RESULT" if not ok else "MISSING"}
        exp_num = next_exp + i
        parent = exp_num - 1
        append_log_md(phase_id, vid, result, exp_num, parent)
        append_results_tsv(phase_id, vid, result, exp_num, parent)
        try: save_sample_png(phase_id, vid)
        except Exception as e: log(phase_id, f"  png warning: {e}")

    log(phase_id, f"All variants done in {(time.time() - overall_t0)/60:.1f} min")
    write_phase_report(phase_id, variants)
    log(phase_id, f"=== {phase_id} END ===")


if __name__ == "__main__":
    main()
