"""Phase runner: chạy ME0→ME1→ME2→ME3 tuần tự, log kết quả."""
import sys, json, time, subprocess
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
WS  = LAB / "workspace"
LOG = LAB / "mapelites.log"
PYTHON = r"C:\Users\Acer\anaconda3\envs\pcgnn\python.exe"
RUNNER = str(WS / "run_mapelites.py")

VARIANTS = ["ME0", "ME1", "ME2", "ME3"]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=== MAP-Elites phase START ===")
log(f"Variants: {VARIANTS}")

results = {}
for variant in VARIANTS:
    out_dir = WS / f"expme-{variant}"
    log(f">>> Starting {variant}")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, RUNNER, variant, str(out_dir)],
        cwd=str(LAB.parent),
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    elapsed = time.time() - t0

    # Save stdout log
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        log(f"<<< {variant} CRASH in {elapsed:.0f}s")
        log(f"    stderr tail: {proc.stderr[-300:]}")
        results[variant] = {"status": "crash", "elapsed_s": elapsed}
        continue

    result_path = out_dir / "result.json"
    if result_path.exists():
        r = json.loads(result_path.read_text())
        compound = r.get("compound", 0)
        coverage = r.get("archive_coverage", "?")
        non_l    = r["summary"].get("non_l_pattern_rate", 0)
        solv     = r["summary"].get("solvability_rate", 0)
        log(f"<<< {variant} OK in {elapsed:.0f}s  compound={compound:.3f}  non_l={non_l:.2f}  solv={solv:.2f}  archive={coverage}")
        results[variant] = {"status": "ok", "elapsed_s": elapsed, **r}
    else:
        log(f"<<< {variant} done but no result.json")
        results[variant] = {"status": "no_result", "elapsed_s": elapsed}

# ─── summary ──────────────────────────────────────────────────────────────────
log("")
log("=== MAP-Elites Summary ===")
log(f"{'Variant':<8} {'Compound':>9} {'Non-L':>8} {'Solv':>7} {'Diag':>6} {'Empty':>6} {'Archive':>10} {'Time':>8}")
log("-" * 75)

best_variant = None
best_compound = -1
for v in VARIANTS:
    r = results.get(v, {})
    if r.get("status") != "ok":
        log(f"{v:<8}  CRASH/SKIP")
        continue
    c   = r.get("compound", 0)
    s   = r["summary"]
    non_l = s.get("non_l_pattern_rate", 0)
    solv  = s.get("solvability_rate", 0)
    diag  = s.get("diagonal_stripes_count", 0)
    empty = s.get("near_empty_count", 0)
    cov   = r.get("archive_coverage", "?")
    t     = r.get("elapsed_s", 0)
    log(f"{v:<8}  {c:>9.3f}  {non_l:>8.2f}  {solv:>7.2f}  {diag:>6}  {empty:>6}  {cov:>10}  {t:>6.0f}s")
    if c > best_compound:
        best_compound = c
        best_variant = v

log("")
log(f"Best variant : {best_variant}  (compound={best_compound:.3f})")
log(f"V10 baseline : compound=0.679")
log(f"V10 baseline : non_l=0.70  solv=0.98  diag=0  empty=1")
log("")
log("=== MAP-Elites phase END ===")

# Save summary JSON
summary_path = LAB / "mapelites_summary.json"
summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
log(f"Summary: {summary_path}")
