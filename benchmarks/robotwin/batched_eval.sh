#!/usr/bin/env bash
# Split topology: dedicated inference GPU(s) + simulation-only GPU(s).
#
#   Infer cards: 1 encoder + 1 batched_server each (no SAPIEN).
#   Sim cards:   n RoboTwin processes each, all talking to their assigned infer.
#   Total parallel envs = (#sim GPUs) * n-sims.
#
# Slot ids are global on each infer server (0 .. n_slots-1). Sim GPUs are
# partitioned across infer GPUs as evenly as possible.
#
# Defaults for this cluster (override freely):
#   1 infer on GPU 2 + 5 sim on GPUs 3-7, n-sims=4  → 20 parallel envs.
#
# Usage:
#   bash benchmarks/robotwin/batched_eval.sh \
#     -d /path/to/ckpt -m both -n smoke \
#     adjust_bottle beat_block_hammer
#
#   # explicit split:
#   bash benchmarks/robotwin/batched_eval.sh -d CKPT -m both -n full \
#     --infer-gpus 2 --sim-gpus 3,4,5,6,7 --n-sims 4 all
#
#   # shorthand: first K of --gpus are infer, rest are sim
#   bash benchmarks/robotwin/batched_eval.sh -d CKPT -m both -n full \
#     --gpus 2,3,4,5,6,7 --n-infer 1 --n-sims 4 all
#
#   # colocate infer+sim on GPU 2 (old-style), + sims on GPU 3; 6 envs total:
#   bash benchmarks/robotwin/batched_eval.sh -d CKPT -m both -n full \
#     --infer-gpus 2 --sim-gpus 2,3 --n-sims 3 all
#
# Env:
#   ROBOTWIN_PATH, ROBOTWIN_PYTHON (or ROBOTWIN_ENV, default RoboTwin)
#   CONDA_ENV   openwam python for inference + encoder (default openwam)
#   PORT_BASE   first inference port (default 9202). Infer ordinal i uses
#               PORT_BASE+10*i and the next port for its encoder.
#   WAN_PATH    UMT5 weights (default: Motus Wan2.2-TI2V-5B)
#   ENCODER_DEVICE  device for UMT5 (default cuda:0 = same physical GPU as infer;
#                   CVD masks each infer card to logical 0. Use cpu to free ~16GB.)
#   MAX_INFER_BATCH max slots per generate_batch (default 2; 0 = unlimited)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

# Task list lives in multi_eval.sh. Pull only the array, do not execute it.
eval "$(sed -n '/^ROBOTWIN_ALL_TASKS=(/,/^)/p' "${SCRIPT_DIR}/multi_eval.sh")"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash benchmarks/robotwin/batched_eval.sh -d <ckpt> -m <mode> -n <name> \
      [gpu options] [options] <tasks...|all>

Required:
  -d, --ckpt-dir   OpenWAM checkpoint directory
  -m, --mode       demo_clean | demo_randomized | both
  -n, --name       label used in log directory names

GPU split (pick one style):
  --infer-gpus     physical ids for inference (+ encoder), e.g. 2
  --sim-gpus       physical ids for simulation only, e.g. 3,4,5,6,7
  --gpus           all ids; use with --n-infer K (first K = infer, rest = sim)
  --n-infer        when using --gpus: how many leading ids are infer (default 1)
  --n-sims         simulation processes per sim GPU (default 4)

  Defaults if none given: infer=2, sim=3,4,5,6,7, n-sims=4

Other:
  --seed           RoboTwin eval seed (default 0)
  --skip-file      lines of mode:task to leave out
  --port-base      first inference port (default 9202)
  --wan-path       UMT5 weights dir
  --resume-from    resume from a prior batched_eval log dir (reuses that dir).
                   Per mode×task: skip if already at --resume-target episodes;
                   else continue at last_seed+1 with prior success counters.
                   GPU / multi-sim scheduling is identical to a fresh run.
  --resume-target  total episodes per task when resuming (default 100)
  --sequential     force width=1 (one sim client at a time; optional debug)

Tasks: names, a comma list, "all", or a file (one task per line).
  With --resume-from and no tasks given, defaults to "all".
EOF
}

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }

CKPT_DIR="" TASK_CONFIG="" POLICY_NAME=""
GPUS="${GPUS:-}"
INFER_GPUS="${INFER_GPUS:-}"
SIM_GPUS="${SIM_GPUS:-}"
N_INFER="${N_INFER:-1}"
N_SIMS="${N_SIMS:-4}"
PORT_BASE="${PORT_BASE:-9202}"
ROBOTWIN_SEED="${ROBOTWIN_SEED:-0}"
SKIP_FILE=""
RESUME_FROM="${RESUME_FROM:-}"
RESUME_TARGET="${RESUME_TARGET:-100}"
SEQUENTIAL="${SEQUENTIAL:-0}"
WAN_PATH="${WAN_PATH:-/data/zixian_guo/projects/haoming/project/Motus/pretrained_models/Wan2.2-TI2V-5B}"
CONDA_ENV="${CONDA_ENV:-openwam}"
# Default: colocate T5 with the infer card (CUDA_VISIBLE_DEVICES=${infer_gpu} → cuda:0).
ENCODER_DEVICE="${ENCODER_DEVICE:-cuda:0}"
# Peak denoise with torch.compile leaves ~80GB resident; 4-wide OOM'd a 96GB card.
MAX_INFER_BATCH="${MAX_INFER_BATCH:-2}"

while (( $# > 0 )); do
    case "$1" in
        -d|--ckpt-dir)   CKPT_DIR="$2"; shift 2 ;;
        -m|--mode)       TASK_CONFIG="$2"; shift 2 ;;
        -n|--name)       POLICY_NAME="$2"; shift 2 ;;
        --gpus)          GPUS="$2"; shift 2 ;;
        --infer-gpus)    INFER_GPUS="$2"; shift 2 ;;
        --sim-gpus)      SIM_GPUS="$2"; shift 2 ;;
        --n-infer)       N_INFER="$2"; shift 2 ;;
        --n-sims)        N_SIMS="$2"; shift 2 ;;
        --port-base)     PORT_BASE="$2"; shift 2 ;;
        --seed)          ROBOTWIN_SEED="$2"; shift 2 ;;
        --skip-file)     SKIP_FILE="$2"; shift 2 ;;
        --wan-path)      WAN_PATH="$2"; shift 2 ;;
        --resume-from)   RESUME_FROM="$2"; shift 2 ;;
        --resume-target) RESUME_TARGET="$2"; shift 2 ;;
        --sequential)    SEQUENTIAL=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)               break ;;
    esac
done

# Resume can recover -d/-m/-n from the log directory when omitted.
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_FROM="$(cd "${RESUME_FROM}" && pwd)"
    [[ -d "${RESUME_FROM}" ]] || { echo "[ERROR] --resume-from not a directory: ${RESUME_FROM}" >&2; exit 1; }
    [[ "${RESUME_TARGET}" =~ ^[0-9]+$ ]] && (( RESUME_TARGET >= 1 )) || {
        echo "[ERROR] --resume-target must be >= 1" >&2; exit 1; }
    # Same task grid as a fresh full eval unless the user names a subset.
    (( $# > 0 )) || set -- all
fi

if [[ -z "${RESUME_FROM}" ]]; then
    [[ -n "${CKPT_DIR}" && -n "${TASK_CONFIG}" && -n "${POLICY_NAME}" ]] || {
        echo "[ERROR] Need -d, -m, -n (or --resume-from)" >&2; usage; exit 1; }
fi

# Resolve infer / sim GPU lists.
if [[ -n "${INFER_GPUS}" || -n "${SIM_GPUS}" ]]; then
    [[ -n "${INFER_GPUS}" && -n "${SIM_GPUS}" ]] || {
        echo "[ERROR] Pass both --infer-gpus and --sim-gpus, or use --gpus/--n-infer" >&2
        exit 1
    }
elif [[ -n "${GPUS}" ]]; then
    IFS=',' read -r -a _all <<< "${GPUS}"
    [[ "${N_INFER}" =~ ^[0-9]+$ ]] && (( N_INFER >= 1 )) || {
        echo "[ERROR] --n-infer must be >= 1" >&2; exit 1; }
    (( N_INFER < ${#_all[@]} )) || {
        echo "[ERROR] --n-infer=${N_INFER} needs at least one sim GPU in --gpus=${GPUS}" >&2
        exit 1
    }
    INFER_GPUS="$(IFS=','; echo "${_all[*]:0:N_INFER}")"
    SIM_GPUS="$(IFS=','; echo "${_all[*]:N_INFER}")"
else
    # Cluster default: 1 infer + 5 sim, n=4.
    INFER_GPUS="${INFER_GPUS:-2}"
    SIM_GPUS="${SIM_GPUS:-3,4,5,6,7}"
fi

# Parse resume plans early (needs python; use system/openwam later for servers).
RESUME_PLAN_FILE=""
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_PLAN_FILE="$(mktemp "${TMPDIR:-/tmp}/batched_eval_resume.XXXXXX.jsonl")"
    _py_resume="${RESUME_PYTHON:-python3}"
    if [[ -x "/data/anaconda3/envs/openwam/bin/python" ]]; then
        _py_resume="/data/anaconda3/envs/openwam/bin/python"
    fi
    # Full plan for the log dir (task filter applied later via the normal TASKS grid).
    "${_py_resume}" "${SCRIPT_DIR}/resume_from_logs.py" "${RESUME_FROM}" \
        --target "${RESUME_TARGET}" \
        --base-seed "${ROBOTWIN_SEED}" \
        > "${RESUME_PLAN_FILE}"
    # Recover missing -d/-m/-n from summary line.
    _summary_json="$("${_py_resume}" -c 'import json,sys
p=sys.argv[1]
last=None
for line in open(p):
    o=json.loads(line)
    if o.get("_summary"): last=o
print(json.dumps(last or {}))' "${RESUME_PLAN_FILE}")"
    if [[ -z "${CKPT_DIR}" ]]; then
        CKPT_DIR="$(printf '%s' "${_summary_json}" | "${_py_resume}" -c 'import json,sys; print(json.load(sys.stdin).get("ckpt_dir") or "")')"
    fi
    if [[ -z "${POLICY_NAME}" ]]; then
        POLICY_NAME="$(printf '%s' "${_summary_json}" | "${_py_resume}" -c 'import json,sys; print(json.load(sys.stdin).get("policy_name") or "")')"
    fi
    if [[ -z "${TASK_CONFIG}" ]]; then
        TASK_CONFIG="$(printf '%s' "${_summary_json}" | "${_py_resume}" -c 'import json,sys; print(json.load(sys.stdin).get("modes") or "both")')"
    fi
    echo "[batched_eval] resume-from=${RESUME_FROM} target=${RESUME_TARGET} recovered ckpt=${CKPT_DIR} name=${POLICY_NAME} modes=${TASK_CONFIG}"
fi

MODES=()
IFS=',' read -ra _mode_parts <<< "${TASK_CONFIG}"
for _mode in "${_mode_parts[@]}"; do
    _mode="$(trim "${_mode}")"
    case "${_mode}" in
        both|all_modes) MODES+=(demo_clean demo_randomized) ;;
        demo_clean) MODES+=(demo_clean) ;;
        demo_randomized|demo_randomize) MODES+=(demo_randomized) ;;
        "") ;;
        *) echo "[ERROR] Invalid mode: ${_mode} (want demo_clean, demo_randomized, or both)" >&2; exit 1 ;;
    esac
done
_deduped=()
for _mode in "${MODES[@]}"; do
    _seen=0
    for _have in "${_deduped[@]+"${_deduped[@]}"}"; do
        [[ "${_have}" == "${_mode}" ]] && { _seen=1; break; }
    done
    (( _seen )) || _deduped+=("${_mode}")
done
MODES=("${_deduped[@]}")
(( ${#MODES[@]} > 0 )) || { echo "[ERROR] No modes in -m ${TASK_CONFIG}" >&2; exit 1; }
MODE_LABEL="$(IFS=+; echo "${MODES[*]}")"
[[ -n "${CKPT_DIR}" && -d "${CKPT_DIR}" ]] || { echo "[ERROR] ckpt not found: ${CKPT_DIR}" >&2; exit 1; }
[[ -n "${POLICY_NAME}" ]] || { echo "[ERROR] missing -n / policy name" >&2; exit 1; }
[[ "${N_SIMS}" =~ ^[0-9]+$ ]] && (( N_SIMS >= 1 )) || { echo "[ERROR] --n-sims must be >= 1" >&2; exit 1; }
if [[ -z "${RESUME_FROM}" ]]; then
    (( $# > 0 )) || { echo "[ERROR] No tasks specified." >&2; usage; exit 1; }
fi

ROBOTWIN_PATH="${ROBOTWIN_PATH:-${ROOT}/third_party/RoboTwin}"
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 1; }
export ROBOTWIN_PATH

if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    if [[ -x "/data/anaconda3/envs/${ROBOTWIN_ENV:-RoboTwin}/bin/python" ]]; then
        ROBOTWIN_PYTHON="/data/anaconda3/envs/${ROBOTWIN_ENV:-RoboTwin}/bin/python"
    fi
fi
[[ -n "${ROBOTWIN_PYTHON:-}" && -x "${ROBOTWIN_PYTHON}" ]] || {
    echo "[ERROR] Set ROBOTWIN_PYTHON to the RoboTwin env python." >&2; exit 1; }
export ROBOTWIN_PYTHON

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
OPENWAM_PYTHON="$(command -v python)"

IFS=',' read -r -a INFER_ARR <<< "${INFER_GPUS}"
IFS=',' read -r -a SIM_ARR <<< "${SIM_GPUS}"
N_INFER_GPU="${#INFER_ARR[@]}"
N_SIM_GPU="${#SIM_ARR[@]}"
(( N_INFER_GPU >= 1 && N_SIM_GPU >= 1 )) || {
    echo "[ERROR] Need >=1 infer GPU and >=1 sim GPU (infer=${INFER_GPUS} sim=${SIM_GPUS})" >&2
    exit 1
}

# Overlap is allowed (colocate): same physical GPU can host infer + sim, as in
# the old N-card×n-sims layout. Warn so it is intentional.
declare -A _infer_set=() _sim_set=()
for g in "${INFER_ARR[@]}"; do
    g="$(trim "${g}")"; [[ -n "${g}" ]] || continue
    _infer_set[$g]=1
done
for g in "${SIM_ARR[@]}"; do
    g="$(trim "${g}")"; [[ -n "${g}" ]] || continue
    _sim_set[$g]=1
done
_colocated=()
for g in "${!_infer_set[@]}"; do
    [[ -n "${_sim_set[$g]:-}" ]] && _colocated+=("${g}")
done
if (( ${#_colocated[@]} > 0 )); then
    echo "[batched_eval] WARNING: colocating infer+sim on GPU(s): $(IFS=','; echo "${_colocated[*]}")" >&2
fi

WIDTH=$(( N_SIM_GPU * N_SIMS ))
if [[ "${SEQUENTIAL}" == "1" ]]; then
    # Optional debug: one in-flight sim client (does not affect seed continuity).
    if (( WIDTH != 1 )); then
        echo "[batched_eval] --sequential: forcing width=1 (was n_sims=${N_SIMS} x ${N_SIM_GPU} sim GPUs)"
    fi
    N_SIMS=1
    SIM_GPUS="${SIM_ARR[0]}"
    SIM_ARR=("${SIM_ARR[0]}")
    N_SIM_GPU=1
    WIDTH=1
fi

# Partition sim GPUs across infer GPUs (contiguous chunks, as even as possible).
# SIM_OWNER[j] = infer ordinal that owns SIM_ARR[j]
# INFER_SIM_COUNT[i] / INFER_SIM_START[i] describe the chunk for infer i.
SIM_OWNER=()
INFER_SIM_COUNT=()
INFER_SIM_START=()
base=$(( N_SIM_GPU / N_INFER_GPU ))
rem=$(( N_SIM_GPU % N_INFER_GPU ))
_cursor=0
for (( i=0; i<N_INFER_GPU; i++ )); do
    _cnt=${base}
    (( i < rem )) && _cnt=$(( _cnt + 1 ))
    INFER_SIM_START[i]=${_cursor}
    INFER_SIM_COUNT[i]=${_cnt}
    for (( j=0; j<_cnt; j++ )); do
        SIM_OWNER[_cursor]=${i}
        _cursor=$(( _cursor + 1 ))
    done
done

# Per-sim-GPU: which infer ordinal, and the first global slot id on that infer.
SIM_INFER_ORD=()
SIM_SLOT_BASE=()
for (( j=0; j<N_SIM_GPU; j++ )); do
    i=${SIM_OWNER[j]}
    SIM_INFER_ORD[j]=${i}
    # slot base = (offset of this sim GPU within its infer's chunk) * N_SIMS
    local_offset=$(( j - INFER_SIM_START[i] ))
    SIM_SLOT_BASE[j]=$(( local_offset * N_SIMS ))
done

TASKS=()
for inp in "$@"; do
    if [[ "${inp}" == "all" ]]; then
        TASKS+=("${ROBOTWIN_ALL_TASKS[@]}")
        continue
    fi
    if [[ -f "${inp}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"
            [[ -n "${line}" ]] && TASKS+=("${line}")
        done < "${inp}"
        continue
    fi
    IFS=',' read -ra parts <<< "${inp}"
    for task in "${parts[@]}"; do
        task="$(trim "${task}")"
        [[ -n "${task}" ]] && TASKS+=("${task}")
    done
done
(( ${#TASKS[@]} > 0 )) || { echo "[ERROR] No tasks resolved." >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RESUME_FROM}" ]]; then
    LOG_DIR="${RESUME_FROM}"
    mkdir -p "${LOG_DIR}"
    echo "[batched_eval] appending logs under resume dir ${LOG_DIR}"
else
    LOG_DIR="${ROOT}/logs/batched_eval_${POLICY_NAME}_${MODE_LABEL}_${STAMP}"
    mkdir -p "${LOG_DIR}"
fi

pids=()
cleanup() {
    trap - INT TERM EXIT
    echo "[batched_eval] stopping ${#pids[@]} processes"
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    [[ -n "${RESUME_PLAN_FILE:-}" && -f "${RESUME_PLAN_FILE:-}" ]] && rm -f "${RESUME_PLAN_FILE}" || true
}
trap cleanup INT TERM EXIT

wait_ws() {
    local port="$1"
    local i
    for i in $(seq 1 300); do
        if "${OPENWAM_PYTHON}" - <<PY
import sys
sys.path.insert(0, "${ROOT}")
from benchmarks.utils.transport import WSPolicyClient
try:
    msg = WSPolicyClient("ws://127.0.0.1:${port}", timeout=2).ping()
except Exception:
    sys.exit(1)
sys.exit(0 if msg.get("type") == "pong" else 1)
PY
        then
            return 0
        fi
        sleep 2
    done
    echo "[ERROR] inference port ${port} did not become healthy" >&2
    return 1
}

export PYTHONUNBUFFERED=1
export ROBOTWIN_SEED

echo "[batched_eval] topology=split infer_gpus=${INFER_GPUS} sim_gpus=${SIM_GPUS} n_sims=${N_SIMS} width=${WIDTH} modes=${MODE_LABEL} tasks=${#TASKS[@]} seed=${ROBOTWIN_SEED} logs=${LOG_DIR}"
for (( i=0; i<N_INFER_GPU; i++ )); do
    echo "[batched_eval]   infer[${i}] gpu=${INFER_ARR[i]} owns ${INFER_SIM_COUNT[i]} sim GPU(s) -> n_slots=$(( INFER_SIM_COUNT[i] * N_SIMS ))"
done

port_busy() {
    local port="$1"
    "${OPENWAM_PYTHON}" - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

for (( i=0; i<N_INFER_GPU; i++ )); do
    infer_port=$(( PORT_BASE + i * 10 ))
    enc_port=$(( infer_port + 1 ))
    if ! port_busy "${infer_port}" || ! port_busy "${enc_port}"; then
        echo "[ERROR] port ${infer_port} or ${enc_port} is already in use. Pass --port-base." >&2
        exit 1
    fi
done

for (( i=0; i<N_INFER_GPU; i++ )); do
    gpu="${INFER_ARR[i]}"
    infer_port=$(( PORT_BASE + i * 10 ))
    enc_port=$(( infer_port + 1 ))
    n_slots=$(( INFER_SIM_COUNT[i] * N_SIMS ))
    (( n_slots >= 1 )) || {
        echo "[ERROR] infer gpu ${gpu} was assigned 0 sim GPUs (uneven split with too many --n-infer?)" >&2
        exit 1
    }
    if [[ "${ENCODER_DEVICE}" == cpu* ]]; then
        CUDA_VISIBLE_DEVICES="" "${OPENWAM_PYTHON}" "${SCRIPT_DIR}/encoder_server.py" \
            --port "${enc_port}" \
            --ckpt-dir "${CKPT_DIR}" \
            --wan-path "${WAN_PATH}" \
            --device "${ENCODER_DEVICE}" \
            > "${LOG_DIR}/encoder_gpu${gpu}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${gpu}" "${OPENWAM_PYTHON}" "${SCRIPT_DIR}/encoder_server.py" \
            --port "${enc_port}" \
            --ckpt-dir "${CKPT_DIR}" \
            --wan-path "${WAN_PATH}" \
            --device "${ENCODER_DEVICE}" \
            > "${LOG_DIR}/encoder_gpu${gpu}.log" 2>&1 &
    fi
    pids+=("$!")
    # expandable_segments reduces fragmentation after large failed batches.
    CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        "${OPENWAM_PYTHON}" "${SCRIPT_DIR}/batched_server.py" \
        --ckpt-dir "${CKPT_DIR}" \
        --port "${infer_port}" \
        --encoder-port "${enc_port}" \
        --n-slots "${n_slots}" \
        --max-batch "${MAX_INFER_BATCH}" \
        --device cuda \
        > "${LOG_DIR}/infer_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "[batched_eval] infer gpu ${gpu}: encoder :${enc_port} (${ENCODER_DEVICE})  inference :${infer_port}  n_slots=${n_slots} max_batch=${MAX_INFER_BATCH}"
done

for (( i=0; i<N_INFER_GPU; i++ )); do
    wait_ws "$(( PORT_BASE + i * 10 ))"
done
echo "[batched_eval] all inference servers healthy"

# One queue across modes. A free slot always pulls the next job.
declare -A SKIP_JOBS=()
if [[ -n "${SKIP_FILE}" ]]; then
    [[ -f "${SKIP_FILE}" ]] || { echo "[ERROR] skip file not found: ${SKIP_FILE}" >&2; exit 1; }
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="$(trim "${line%%#*}")"
        [[ -n "${line}" ]] || continue
        [[ "${line}" == *:* ]] || { echo "[ERROR] skip line must be mode:task, got: ${line}" >&2; exit 1; }
        SKIP_JOBS["${line}"]=1
    done < "${SKIP_FILE}"
fi

job_mode=()
job_task=()
job_st_seed=()
job_resume_done=()
job_resume_suc=()
job_log=()
skipped=0
resume_hit=0
fresh_hit=0

# mode:task -> resume fields (only jobs that already have a log in RESUME_FROM).
declare -A R_DONE=() R_SUC=() R_NEXT=() R_LOG=() R_COMPLETE=()
if [[ -n "${RESUME_FROM}" ]]; then
    [[ -n "${RESUME_PLAN_FILE}" && -f "${RESUME_PLAN_FILE}" ]] || {
        echo "[ERROR] resume plan missing" >&2; exit 1; }
    while IFS=$'\t' read -r _rk _rdone _rsuc _rnext _rcomplete _rlog || [[ -n "${_rk}" ]]; do
        [[ -n "${_rk}" ]] || continue
        R_DONE["${_rk}"]="${_rdone}"
        R_SUC["${_rk}"]="${_rsuc}"
        R_NEXT["${_rk}"]="${_rnext}"
        R_COMPLETE["${_rk}"]="${_rcomplete}"
        R_LOG["${_rk}"]="${_rlog}"
    done < <("${OPENWAM_PYTHON}" -c '
import json, sys
for line in open(sys.argv[1]):
    o = json.loads(line)
    if o.get("_summary"):
        continue
    key = "{}:{}".format(o["mode"], o["task"])
    print("\t".join([
        key,
        str(o["done"]),
        str(o["suc"]),
        str(o["next_seed"]),
        "1" if o.get("skip") or o.get("complete") else "0",
        o.get("log") or "",
    ]))
' "${RESUME_PLAN_FILE}")
fi

# Same mode×task grid as a fresh run; overlay resume state when present.
for TASK_CONFIG in "${MODES[@]}"; do
    for task in "${TASKS[@]}"; do
        _key="${TASK_CONFIG}:${task}"
        if [[ -n "${SKIP_JOBS[${_key}]:-}" ]]; then
            skipped=$(( skipped + 1 ))
            continue
        fi
        if [[ -n "${R_COMPLETE[${_key}]:-}" ]]; then
            if [[ "${R_COMPLETE[${_key}]}" == "1" ]]; then
                skipped=$(( skipped + 1 ))
                continue
            fi
            job_mode+=("${TASK_CONFIG}")
            job_task+=("${task}")
            job_st_seed+=("${R_NEXT[${_key}]}")
            job_resume_done+=("${R_DONE[${_key}]}")
            job_resume_suc+=("${R_SUC[${_key}]}")
            job_log+=("${R_LOG[${_key}]}")
            resume_hit=$(( resume_hit + 1 ))
        else
            # Never started in the prior dir: launch fresh like a from-scratch eval.
            job_mode+=("${TASK_CONFIG}")
            job_task+=("${task}")
            job_st_seed+=("")
            job_resume_done+=("")
            job_resume_suc+=("")
            job_log+=("")
            fresh_hit=$(( fresh_hit + 1 ))
        fi
    done
done
if [[ -n "${RESUME_FROM}" ]]; then
    echo "[batched_eval] resume queue ${#job_task[@]} jobs (resume=${resume_hit} fresh=${fresh_hit} skip=${skipped}), target=${RESUME_TARGET}, width=${WIDTH}"
else
    echo "[batched_eval] queue ${#job_task[@]} jobs, skipped ${skipped}"
fi
job_total="${#job_task[@]}"
(( job_total > 0 )) || { echo "[ERROR] No jobs to run (all complete or filtered)." >&2; exit 1; }
job_next=0
slot_pid=()
for (( k=0; k<WIDTH; k++ )); do
    slot_pid[k]=0
done

launch_into_slot() {
    local k="$1"
    local mode="${job_mode[$job_next]}"
    local task="${job_task[$job_next]}"
    local st_seed="${job_st_seed[$job_next]:-}"
    local resume_done="${job_resume_done[$job_next]:-}"
    local resume_suc="${job_resume_suc[$job_next]:-}"
    local prior_log="${job_log[$job_next]:-}"
    job_next=$(( job_next + 1 ))
    local sim_i=$(( k / N_SIMS ))
    local local_slot=$(( k % N_SIMS ))
    local sim_gpu="${SIM_ARR[$sim_i]}"
    local infer_ord="${SIM_INFER_ORD[$sim_i]}"
    local infer_gpu="${INFER_ARR[$infer_ord]}"
    local infer_port=$(( PORT_BASE + infer_ord * 10 ))
    # Global slot id on that infer server (must be unique among its clients).
    local global_slot=$(( SIM_SLOT_BASE[sim_i] + local_slot ))
    local log="${LOG_DIR}/${mode}_${task}_gpu${sim_gpu}_slot${global_slot}.log"
    # Resume: append into the original task log so Success-rate history stays contiguous.
    if [[ -n "${prior_log}" ]]; then
        log="${prior_log}"
    fi
    local extra_msg=""
    if [[ -n "${st_seed}" ]]; then
        extra_msg=" resume done=${resume_done}/${RESUME_TARGET} suc=${resume_suc} next_seed=${st_seed}"
    fi
    echo "[batched_eval] ${job_next}/${job_total} ${mode} ${task} -> sim_gpu ${sim_gpu} slot ${global_slot} (infer gpu ${infer_gpu} :${infer_port})${extra_msg}"
    (
        export ROBOTWIN_SLOT="${global_slot}"
        if [[ -n "${st_seed}" ]]; then
            export ROBOTWIN_ST_SEED="${st_seed}"
            export ROBOTWIN_RESUME_DONE="${resume_done}"
            export ROBOTWIN_RESUME_SUC="${resume_suc}"
            export ROBOTWIN_TARGET_EPISODES="${RESUME_TARGET}"
            # Avoid ROBOTWIN_TEST_NUM shadowing the cumulative target.
            unset ROBOTWIN_TEST_NUM || true
            {
                echo ""
                echo "[batched_eval] ===== resume $(date -Is) st_seed=${st_seed} done=${resume_done} suc=${resume_suc} target=${RESUME_TARGET} ====="
            } >> "${log}"
        fi
        bash "${SCRIPT_DIR}/single_eval.sh" \
            "${task}" "${mode}" "${POLICY_NAME}" "${sim_gpu}" "${infer_port}" "127.0.0.1" \
            >> "${log}" 2>&1
    ) &
    slot_pid[k]=$!
    pids+=("$!")
}

running=0
for (( k=0; k<WIDTH && job_next<job_total; k++ )); do
    launch_into_slot "${k}"
    running=$(( running + 1 ))
done

fail=0
while (( running > 0 )); do
    sleep 1
    for (( k=0; k<WIDTH; k++ )); do
        pid="${slot_pid[k]}"
        (( pid == 0 )) && continue
        if [[ -d "/proc/${pid}" ]]; then
            state="$(sed -n 's/.*) //p' "/proc/${pid}/stat" 2>/dev/null | awk '{print $1}')"
            [[ "${state}" == "Z" ]] || continue
        fi
        status=0
        # 127 = not a child (already reaped by cleanup/race); ignore as failure.
        if ! wait "${pid}" 2>/dev/null; then
            status=$?
        fi
        if (( status == 127 )); then
            status=0
        fi
        slot_pid[k]=0
        running=$(( running - 1 ))
        if (( status != 0 )); then
            fail=1
            echo "[ERROR] sim pid ${pid} exited ${status}. Continuing remaining queue. See ${LOG_DIR}" >&2
        fi
        if (( job_next < job_total )); then
            launch_into_slot "${k}"
            running=$(( running + 1 ))
        else
            sim_i=$(( k / N_SIMS ))
            local_slot=$(( k % N_SIMS ))
            global_slot=$(( SIM_SLOT_BASE[sim_i] + local_slot ))
            echo "[batched_eval] sim_gpu ${SIM_ARR[$sim_i]} slot ${global_slot} idle (${job_next}/${job_total} dispatched)"
        fi
    done
done
if (( fail )); then
    exit 1
fi

echo "[batched_eval] done. logs: ${LOG_DIR}"
trap - INT TERM EXIT
cleanup
