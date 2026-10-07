#!/usr/bin/env bash
# Migrate OpenWAM RoboTwin2.0 (+ T5 caches) to DFS, then cut over local path
# to a symlink so configs keep working without edits.
#
# Layout (same as OpenWAM assets/benchmark_data):
#   SRC: <OpenWAM>/assets/benchmark_data/robotwin2.0
#   DST: /dfs/dataset/data/assets/benchmark_data/robotwin2.0
#
# Stages:
#   1) rsync mirror (safe; leaves SRC intact)
#   2) verify file counts + total bytes
#   3) cutover: rename SRC aside, symlink DST -> original path, then delete aside
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-${ROOT}/assets/benchmark_data/robotwin2.0}"
DST_ROOT="${DST_ROOT:-/dfs/dataset/data/assets/benchmark_data}"
DST="${DST:-${DST_ROOT}/robotwin2.0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/migrate_robotwin_to_dfs_${STAMP}.log"
SKIP_DELETE="${SKIP_DELETE:-0}"   # 1 = keep local .migrated_bak after symlink
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}" "${DST_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

echo "===== $(date) RoboTwin → DFS migrate ====="
echo "SRC=${SRC}"
echo "DST=${DST}"
echo "LOG=${LOG}"
echo "SKIP_DELETE=${SKIP_DELETE} DRY_RUN=${DRY_RUN}"

if [[ ! -d "${SRC}" ]]; then
  echo "[ERROR] SRC missing: ${SRC}" >&2
  exit 1
fi
if [[ -L "${SRC}" ]]; then
  echo "[INFO] SRC already a symlink → $(readlink -f "${SRC}" || readlink "${SRC}")"
  echo "Nothing to migrate."
  exit 0
fi

# Refuse if something still looks like an active RoboTwin train.
if pgrep -af 'scripts/train.py' | grep -E 'robotwin|clean2random' >/dev/null 2>&1; then
  echo "[ERROR] RoboTwin-related train.py still running; abort migrate." >&2
  pgrep -af 'scripts/train.py' | grep -E 'robotwin|clean2random' || true
  exit 2
fi

echo "$(date) === stage 1/3: rsync ==="
# --partial keeps incomplete files across restarts; avoid --append-verify on
# first full copy (it re-reads every dest file and is very slow on DFS).
RSYNC_FLAGS=(-aH --numeric-ids --info=stats2,progress2 --partial)
if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi
# Trailing slash: copy contents into DST (create DST).
mkdir -p "${DST}"
rsync "${RSYNC_FLAGS[@]}" "${SRC}/" "${DST}/"
echo "$(date) rsync exit=0"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — skip verify/cutover"
  exit 0
fi

echo "$(date) === stage 2/3: verify ==="
python3 - <<'PY' "${SRC}" "${DST}"
import os, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])

def walk_stats(root: Path):
    n_files = n_dirs = 0
    total = 0
    # sample a few relative paths for existence checks
    samples = []
    for dirpath, dirnames, filenames in os.walk(root):
        n_dirs += 1
        # skip heavy progress noise
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            n_files += 1
            total += st.st_size
            if len(samples) < 32 and (
                fn.endswith((".hdf5", ".pt", ".json", ".npy")) or fn == "seed.txt"
            ):
                samples.append(p.relative_to(root).as_posix())
    return n_files, n_dirs, total, samples

print(f"scanning SRC={src} ...", flush=True)
sf, sd, sb, samples = walk_stats(src)
print(f"scanning DST={dst} ...", flush=True)
df, dd, db, _ = walk_stats(dst)
print(f"SRC files={sf} dirs={sd} bytes={sb}")
print(f"DST files={df} dirs={dd} bytes={db}")
missing = []
for rel in samples:
    if not (dst / rel).is_file():
        missing.append(rel)
if missing:
    print("MISSING samples:", missing[:10])
    raise SystemExit(3)
# Allow tiny dir-count drift from empty dirs; files+bytes must match.
if sf != df or sb != db:
    print("[ERROR] file count or byte total mismatch")
    raise SystemExit(4)
print("verify OK")
PY

echo "$(date) === stage 3/3: cutover (symlink) ==="
BAK="${SRC}.migrated_bak_${STAMP}"
# Same-filesystem rename is instant; then point original path at DFS.
mv "${SRC}" "${BAK}"
ln -s "${DST}" "${SRC}"
echo "symlink: ${SRC} -> ${DST}"
ls -ld "${SRC}"
# Sanity: dataset_dir used by config must resolve.
if [[ ! -d "${SRC}/dataset" ]]; then
  echo "[ERROR] after symlink, ${SRC}/dataset missing — restoring bak" >&2
  rm -f "${SRC}"
  mv "${BAK}" "${SRC}"
  exit 5
fi
# Spot-check one known task path.
if [[ ! -d "${SRC}/dataset/adjust_bottle/aloha-agilex_clean_50" ]]; then
  echo "[WARN] adjust_bottle clean_50 missing under symlink (layout may differ)"
fi

if [[ "${SKIP_DELETE}" == "1" ]]; then
  echo "SKIP_DELETE=1 — keeping ${BAK}"
else
  echo "$(date) deleting local backup ${BAK} to free /data ..."
  rm -rf "${BAK}"
  echo "$(date) local backup deleted"
fi

echo "$(date) ===== migrate done ====="
echo "Local path now: $(ls -ld "${SRC}")"
df -h /data /dfs | head -5 || true
