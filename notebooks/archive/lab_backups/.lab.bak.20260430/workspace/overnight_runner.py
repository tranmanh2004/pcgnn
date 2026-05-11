"""
Overnight queue runner — self-contained.

Runs all BC + fitness penalty variants sequentially, auto-updates lab files,
generates comparison report + visualizations. Only one Bash invocation needed.

Usage:
    python overnight_runner.py
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
PROJECT_ROOT = LAB_ROOT.parent
PYTHON = r"C:\Users\Acer\anaconda3\envs\pcgnn\python.exe"

VARIANTS_TO_RUN = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]
EXP_ID = "ovn"
EXP_NUM_START = 6  # next experiment number after #5 in results.tsv

OVERNIGHT_LOG = LAB_ROOT / "overnight.log"
SUMMARY_JSON = LAB_ROOT / "overnight_summary.json"
SUMMARY_MD = LAB_ROOT / "overnight_summary.md"
COMPARISON_PNG = LAB_ROOT / "overnight_comparison.png"
LAB_LOG_MD = LAB_ROOT / "log.md"
LAB_RESULTS_TSV = LAB_ROOT / "results.tsv"

WALL, FLOOR, PLAYER, ENEMY = 0, 1, 2, 3
TILE_COLORS = {
    WALL:   [0.15, 0.15, 0.15],
    FLOOR:  [0.95, 0.95, 0.95],
    PLAYER: [0.20, 0.80, 0.20],
    ENEMY:  [0.90, 0.20, 0.20],
}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OVERNIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_variant(variant_id):
    out_dir = LAB_DIR / f"expv-{variant_id}-{EXP_ID}"
    log_path = LAB_DIR / f"expv-{variant_id}-{EXP_ID}.log"
    log(f">>> Starting variant {variant_id}")
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                [PYTHON, str(LAB_DIR / "run_variant.py"), variant_id, EXP_ID],
                stdout=f, stderr=subprocess.STDOUT,
                timeout=4500,
            )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"<<< {variant_id} OK in {elapsed:.0f}s")
            return True
        else:
            log(f"<<< {variant_id} FAILED rc={proc.returncode} after {elapsed:.0f}s — see {log_path.name}")
            return False
    except subprocess.TimeoutExpired:
        log(f"<<< {variant_id} TIMEOUT after {(time.time() - t0):.0f}s")
        return False
    except Exception as e:
        log(f"<<< {variant_id} EXCEPTION: {e}")
        return False


def append_log_md(variant_id, result, exp_num, parent_exp):
    """Append entry to .lab/log.md."""
    if not result or result.get("status") != "OK":
        block = (
            f"\n## Experiment {exp_num} — {variant_id} (FAILED)\n"
            f"Branch: research/bc-vector-l-pattern / Type: real / Parent: #{parent_exp}\n"
            f"Status: {result.get('status', 'UNKNOWN') if result else 'NO_RESULT'}\n"
        )
    else:
        h = result.get("turns_histogram", [0]*10)
        hist_str = ", ".join(f"{l}:{c}" for l, c in zip(
            ["0","1","2","3","4","5","6","7","≥8","unsolv"], h))
        block = (
            f"\n## Experiment {exp_num} — {variant_id}\n"
            f"Branch: research/bc-vector-l-pattern / Type: real / Parent: #{parent_exp}\n"
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
            f"Status: keep (logged from overnight runner)\n"
        )
    with open(LAB_LOG_MD, "a", encoding="utf-8") as f:
        f.write(block)


def append_results_tsv(variant_id, result, exp_num, parent_exp):
    """Append row to .lab/results.tsv."""
    if not result or result.get("status") != "OK":
        row = f"{exp_num}\tresearch/bc-vector-l-pattern\t{parent_exp}\t-\t-1\t-\tcrash\t-\t{variant_id}: FAILED\n"
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
        desc = f"{variant_id}: {result.get('description', '')[:80]}"
        row = (
            f"{exp_num}\tresearch/bc-vector-l-pattern\t{parent_exp}\t-\t"
            f"{result['non_l_pattern_rate']:.6f}\t{secondary}\tkeep\t"
            f"{result.get('train_seconds', 0):.1f}\t{desc}\n"
        )
    with open(LAB_RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(row)


def save_variant_sample_png(variant_id, n_show=10):
    """Save 10-map sample PNG for one variant."""
    out_dir = LAB_DIR / f"expv-{variant_id}-{EXP_ID}"
    maps_path = out_dir / "maps.pkl"
    if not maps_path.exists():
        return None
    with open(maps_path, "rb") as f:
        maps = pickle.load(f)
    fig, axs = plt.subplots(1, n_show, figsize=(n_show * 1.6, 1.8))
    for j in range(min(n_show, len(maps))):
        lvl = maps[j]
        h, w = lvl.shape
        img = np.zeros((h, w, 3))
        for tile, color in TILE_COLORS.items():
            img[lvl == tile] = color
        axs[j].imshow(img, interpolation="nearest")
        axs[j].axis("off")
    plt.suptitle(f"Variant {variant_id}", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / "sample.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def save_comparison_png(rows):
    """Save big grid comparing 10 maps from each variant."""
    n_vars = len(rows)
    n_show = 10
    fig, axs = plt.subplots(n_vars, n_show, figsize=(n_show * 1.5, n_vars * 1.7))
    for i, r in enumerate(rows):
        vid = r["variant_id"]
        out_dir = LAB_DIR / f"expv-{vid}-{EXP_ID}"
        maps_path = out_dir / "maps.pkl"
        if not maps_path.exists():
            for j in range(n_show):
                axs[i, j].text(0.5, 0.5, "MISSING",
                               ha="center", va="center", transform=axs[i, j].transAxes)
                axs[i, j].axis("off")
            continue
        with open(maps_path, "rb") as f:
            maps = pickle.load(f)
        for j in range(n_show):
            lvl = maps[j]
            h, w = lvl.shape
            img = np.zeros((h, w, 3))
            for tile, color in TILE_COLORS.items():
                img[lvl == tile] = color
            axs[i, j].imshow(img, interpolation="nearest")
            axs[i, j].axis("off")
        if r.get("status") == "OK":
            label = (
                f"{vid}\nnon_l={r['non_l_pattern_rate']:.2f} "
                f"solv={r['solvability']:.2f}\n"
                f"diag={r['diagonal_stripes_count']} empty={r['near_empty_count']}"
            )
        else:
            label = f"{vid}\n{r.get('status', 'FAIL')}"
        axs[i, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=70, va="center", ha="right")
    plt.suptitle("Overnight comparison — 10 sample maps per variant", fontsize=12)
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=120, bbox_inches="tight")
    plt.close()


def aggregate_results():
    rows = []
    for vid in VARIANTS_TO_RUN:
        result_path = LAB_DIR / f"expv-{vid}-{EXP_ID}" / "result.json"
        if not result_path.exists():
            rows.append({"variant_id": vid, "status": "MISSING"})
            continue
        try:
            r = json.loads(result_path.read_text())
            r["status"] = "OK"
            rows.append(r)
        except Exception as e:
            rows.append({"variant_id": vid, "status": f"PARSE_ERROR: {e}"})
    return rows


def compute_compound(r):
    if r.get("status") != "OK":
        return -1.0
    n = r.get("n_maps", 100)
    diag_pct = r.get("diagonal_stripes_count", 0) / n
    extreme_pct = (r.get("near_empty_count", 0) + r.get("near_full_count", 0)) / n
    return (
        r.get("non_l_pattern_rate", 0)
        * r.get("solvability", 0)
        * (1 - diag_pct)
        * (1 - extreme_pct)
    )


def write_report(rows):
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2))
    md = ["# Overnight Run Summary\n"]
    md.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"Variants run: {len(VARIANTS_TO_RUN)}\n")

    md.append("\n## Comparison Table (sorted by compound metric)\n")
    md.append("Compound = non_l × solv × (1 − diag%) × (1 − extreme%)\n")
    md.append("\n| Rank | Variant | Description | non_l | solv | diag/100 | empty | full | div | diff | Compound |")
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

    best_ok = [r for r in ranked if r.get("status") == "OK"]
    if best_ok:
        b = best_ok[0]
        md.append(f"\n## Winner: **{b['variant_id']}** (compound={compute_compound(b):.3f})\n")
        md.append(f"- Description: {b.get('description')}\n")
        md.append(f"- non_l_pattern_rate: {b['non_l_pattern_rate']:.4f}\n")
        md.append(f"- solvability: {b['solvability']:.4f}\n")
        md.append(f"- diagonal stripes: {b['diagonal_stripes_count']}/{b['n_maps']} ({100*b['diagonal_stripes_count']/b['n_maps']:.0f}%)\n")
        md.append(f"- near-empty: {b['near_empty_count']}/{b['n_maps']}\n")
        md.append(f"- near-full: {b['near_full_count']}/{b['n_maps']}\n")
        md.append(f"- A* difficulty: {b['astar_diff_mean']:.4f}\n")
        md.append(f"- pairwise A* diversity: {b['astar_div_pairs_mean']:.4f}\n")

    md.append("\n## Turns Histograms\n")
    md.append("| Variant | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥8 | unsolv |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "OK":
            continue
        h = r.get("turns_histogram", [0] * 10)
        md.append(f"| {r['variant_id']} | " + " | ".join(str(x) for x in h) + " |")

    md.append("\n## Files\n")
    md.append(f"- Raw results: `.lab/overnight_summary.json`\n")
    md.append(f"- Comparison image: `.lab/overnight_comparison.png`\n")
    md.append(f"- Per-variant: `.lab/workspace/expv-V*-{EXP_ID}/{{maps.pkl, winner.pkl, result.json, sample.png}}`\n")
    md.append(f"- Progress log: `.lab/overnight.log`\n")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")


def update_config_md(rows):
    """Update .lab/config.md best-so-far if new compound winner."""
    best_ok = [r for r in rows if r.get("status") == "OK"]
    if not best_ok:
        return
    best = max(best_ok, key=compute_compound)
    block = (
        f"\n\n## Phase 2 winner (overnight run, {time.strftime('%Y-%m-%d')})\n\n"
        f"- Variant: **{best['variant_id']}**\n"
        f"- Description: {best.get('description')}\n"
        f"- non_l_pattern_rate: {best['non_l_pattern_rate']:.4f}\n"
        f"- solvability: {best['solvability']:.4f}\n"
        f"- diagonal stripes: {best['diagonal_stripes_count']}/100\n"
        f"- compound: {compute_compound(best):.4f}\n"
    )
    with open(LAB_ROOT / "config.md", "a", encoding="utf-8") as f:
        f.write(block)


def main():
    log(f"=== OVERNIGHT RUN START — {len(VARIANTS_TO_RUN)} variants ===")
    overall_t0 = time.time()

    # Initial THINK entry into log.md
    with open(LAB_LOG_MD, "a", encoding="utf-8") as f:
        f.write(f"\n\n## THINK — Phase 2 overnight queue ({time.strftime('%Y-%m-%d %H:%M')})\n")
        f.write("Convergence signals: BC phase #5 hit target 0.88. New mode collapse identified ")
        f.write("(diagonal stripes 44%, near-empty 34%) via 100-map evaluation.\n")
        f.write("Untested assumptions: \n")
        f.write("  - Whether fitness penalty can stop gaming behavior (V1, V2, V3)\n")
        f.write("  - Whether higher-dim BC (entropy/corridor) breaks stripes (V4, V5)\n")
        f.write("  - Whether reducing turns weight + adding penalty produces balanced result (V6)\n")
        f.write("Invalidation risk: V0 re-run validates that current best is reproducible.\n")
        f.write(f"Queue: {', '.join(VARIANTS_TO_RUN)}. Compound metric.\n")

    # Run each variant
    for i, vid in enumerate(VARIANTS_TO_RUN):
        exp_num = EXP_NUM_START + i
        parent = exp_num - 1 if i > 0 else 5  # V0 parent = #5 winner
        ok = run_variant(vid)
        # Read result and update lab files immediately
        result_path = LAB_DIR / f"expv-{vid}-{EXP_ID}" / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                result["status"] = "OK"
            except Exception as e:
                result = {"status": f"PARSE_ERROR: {e}"}
        else:
            result = {"status": "NO_RESULT"} if not ok else {"status": "MISSING"}
        append_log_md(vid, result, exp_num, parent)
        append_results_tsv(vid, result, exp_num, parent)
        # Save per-variant PNG
        try:
            save_variant_sample_png(vid)
        except Exception as e:
            log(f"  warning: sample.png for {vid} failed: {e}")

    # After all variants
    log(f"All variants complete in {(time.time() - overall_t0)/60:.1f} minutes")
    log("Aggregating results...")
    rows = aggregate_results()
    write_report(rows)
    update_config_md(rows)
    try:
        save_comparison_png(rows)
        log(f"Comparison image: {COMPARISON_PNG}")
    except Exception as e:
        log(f"  warning: comparison png failed: {e}")

    log(f"Report: {SUMMARY_MD}")
    log("=== OVERNIGHT RUN END ===")


if __name__ == "__main__":
    main()
