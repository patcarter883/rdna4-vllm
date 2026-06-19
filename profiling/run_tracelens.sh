#!/usr/bin/env bash
# Analyse vLLM PyTorch profiler traces with TraceLens.
#
# Usage:
#   profiling/run_tracelens.sh <trace_dir>          # analyse all *.pt.trace.json.gz in dir
#   profiling/run_tracelens.sh --out <out_dir> <trace_dir>
#
# Output goes to profiling/tracelens/<trace_dir_basename>/ by default.
# Requires TraceLens installed on the host:
#   uv tool install "git+https://github.com/AMD-AGI/TraceLens.git"
#
# NO GPU required — this is host-side post-processing only.
# Do NOT wrap this in scripts/gpu-lease.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT="$SCRIPT_DIR/tracelens"

usage() { echo "Usage: $0 [--out <dir>] <trace_dir>"; exit 1; }

# Parse args
OUT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_OVERRIDE="$2"; shift 2 ;;
    -*) usage ;;
    *) TRACE_DIR="$1"; shift ;;
  esac
done
[[ -z "${TRACE_DIR:-}" ]] && usage

TRACE_DIR="$(realpath "$TRACE_DIR")"
BASENAME="$(basename "$TRACE_DIR")"
OUTDIR="${OUT_OVERRIDE:-$OUTROOT/$BASENAME}"
mkdir -p "$OUTDIR"

echo "TraceLens: analysing traces in $TRACE_DIR"
echo "           output -> $OUTDIR"
echo

# Find all rank traces
shopt -s nullglob
TRACES=( "$TRACE_DIR"/*.pt.trace.json.gz "$TRACE_DIR"/*.pt.trace.json )
shopt -u nullglob

if [[ ${#TRACES[@]} -eq 0 ]]; then
  echo "ERROR: no *.pt.trace.json[.gz] files in $TRACE_DIR" >&2
  exit 1
fi

echo "Found ${#TRACES[@]} trace file(s)."

# --- Per-rank reports ---
for TRACE in "${TRACES[@]}"; do
  TBASE="$(basename "$TRACE")"
  # Extract a short rank tag from the filename (e.g. rank0 from ..._rank0.*.json.gz)
  RANK_TAG="$(echo "$TBASE" | grep -oP 'rank\d+' | head -1)"
  [[ -z "$RANK_TAG" ]] && RANK_TAG="$(echo "$TBASE" | sed 's/\..*$//')"
  RANKOUT="$OUTDIR/$RANK_TAG"
  mkdir -p "$RANKOUT"

  echo "--- per-rank report: $TBASE -> $RANKOUT ---"
  TraceLens_generate_perf_report_pytorch \
    --profile_json_path "$TRACE" \
    --output_csvs_dir "$RANKOUT" \
    --enable_kernel_summary \
    --topk_ops 30 \
    --topk_roofline_ops 20 \
    2>/dev/null
done

# --- Multi-rank collective report (if >1 trace) ---
if [[ ${#TRACES[@]} -gt 1 ]]; then
  MULTIOUT="$OUTDIR/multi"
  mkdir -p "$MULTIOUT"
  echo
  echo "--- multi-rank collective report -> $MULTIOUT ---"
  TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_glob "$TRACE_DIR/*.pt.trace.json.gz" \
    --world_size "${#TRACES[@]}" \
    --output_csvs_dir "$MULTIOUT" \
    2>/dev/null
fi

# --- Print key summary to stdout ---
echo
echo "==========================================================="
echo " SUMMARY: $(basename "$TRACE_DIR")"
echo "==========================================================="

python3 - "$OUTDIR" <<'EOF'
import sys, csv, os, glob

outdir = sys.argv[1]

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return list(csv.DictReader(f))

# GPU timeline (first rank found)
for tpath in sorted(glob.glob(f"{outdir}/rank*/gpu_timeline.csv")):
    rows = read_csv(tpath)
    rname = os.path.basename(os.path.dirname(tpath))
    print(f"\n[{rname}] GPU timeline:")
    for r in rows:
        print(f"  {r.get('type',''):<30} {float(r.get('time ms',0)):>8.1f} ms  {float(r.get('percent',0)):>6.2f}%")

# Ops by category (first rank)
for tpath in sorted(glob.glob(f"{outdir}/rank*/ops_summary_by_category.csv")):
    rows = read_csv(tpath)
    rname = os.path.basename(os.path.dirname(tpath))
    print(f"\n[{rname}] Ops by category:")
    for r in rows:
        print(f"  {r.get('op category',''):<25} {float(r.get('total_direct_kernel_time_ms',0)):>8.1f} ms  {float(r.get('Percentage (%)',0)):>5.1f}%")
    break  # one rank is enough for category breakdown

# Top 10 kernels (first rank)
for tpath in sorted(glob.glob(f"{outdir}/rank*/kernel_summary.csv")):
    rows = read_csv(tpath)
    rname = os.path.basename(os.path.dirname(tpath))
    print(f"\n[{rname}] Top 10 kernels by time:")
    rows_sorted = sorted(rows, key=lambda r: -float(r.get('Kernel duration (µs)_sum', 0)))
    for r in rows_sorted[:10]:
        kname = r.get('Kernel name', '')[:60]
        parent = r.get('Parent cpu_op', '')[:30]
        dur = float(r.get('Kernel duration (µs)_sum', 0)) / 1000
        pct = float(r.get('Percent of total time (%)', 0))
        print(f"  {pct:5.1f}%  {dur:7.1f} ms  [{parent}]  {kname}")
    break

# Straggler summary (multi-rank)
spath = f"{outdir}/multi/straggler_summary.csv"
if os.path.exists(spath):
    rows = read_csv(spath)
    print("\n[multi] Straggler summary (ranks):")
    for r in rows:
        rank = r.get('rank', '?')
        wait = float(r.get('total_wait_time_us', 0))
        mean_wait = float(r.get('mean_wait_time_us', 0))
        pct_last = float(r.get('pct_arrived_last', 0))
        nccl = float(r.get('total_nccl_dur_us', 0)) / 1000
        print(f"  rank {rank}: total_wait={wait/1000:.1f}ms  mean_wait={mean_wait:.1f}µs  "
              f"arrived_last={pct_last:.1f}%  total_nccl={nccl:.1f}ms")

print()
EOF

echo "Full CSVs: $OUTDIR"
