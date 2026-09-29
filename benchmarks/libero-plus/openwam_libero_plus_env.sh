#!/usr/bin/env bash
# Worker / client env for OpenWAM LIBERO-plus MuJoCo EGL evaluation.
#
# Mirrors FastWAM scripts/setup/fastwam_worker_env.sh:
#   1) NVIDIA user-space EGL via VulkanDrive (no system graphic driver)
#   2) ImageMagick MagickWand via ~/.local/conda/imagemagick/lib_nogl
#      (full Magick lib/ ships Mesa EGL that shadows NVIDIA)
#
# Usage:
#   source benchmarks/libero-plus/openwam_libero_plus_env.sh
#   # then run_smoke.sh / run_eval.sh / single_eval.sh

_OW_LIBERO_PLUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_OW_ROOT="$(cd "${_OW_LIBERO_PLUS_DIR}/../.." && pwd)"
_PI_ROOT="$(cd "${_OW_ROOT}/.." && pwd)"

# Prefer dedicated libero-plus conda python on PATH.
if [[ -x "/data/anaconda3/envs/libero-plus/bin/python" ]]; then
  export PATH="/data/anaconda3/envs/libero-plus/bin:${PATH}"
fi

# NVIDIA Vulkan/EGL user libs (H20/H200 headless).
if [[ -f "${_PI_ROOT}/VulkanDrive/env_nvidia_vulkan.sh" ]]; then
  # shellcheck disable=SC1091
  source "${_PI_ROOT}/VulkanDrive/env_nvidia_vulkan.sh"
elif [[ -f "${_PI_ROOT}/ImageWAM/scripts/flux2/env_nvidia_vulkan.sh" ]]; then
  # shellcheck disable=SC1091
  source "${_PI_ROOT}/ImageWAM/scripts/flux2/env_nvidia_vulkan.sh"
fi

export MAGICK_HOME="${MAGICK_HOME:-$HOME/.local/conda/imagemagick}"
MAGICK_LIB_NOGL="${MAGICK_HOME}/lib_nogl"
MAGICK_LIB="${MAGICK_HOME}/lib"
if [[ -d "${MAGICK_LIB_NOGL}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_VULKAN_LIB_DIR:-$HOME/.local/nvidia/lib64_vulkan}:${MAGICK_LIB_NOGL}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "[openwam_libero_plus_env] WARN: ${MAGICK_LIB_NOGL} missing; Magick GL may shadow NVIDIA EGL" >&2
  if [[ -d "${MAGICK_LIB}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_VULKAN_LIB_DIR:-$HOME/.local/nvidia/lib64_vulkan}:${MAGICK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

# LIBERO-plus checkout (FastWAM/ImageWAM share one tree via symlink).
export LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-${_PI_ROOT}/ImageWAM/third_party/LIBERO-plus}"
export LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/data/anaconda3/envs/libero-plus/bin/python}"
export LIBERO_PLUS_CONFIG_ROOT="${LIBERO_PLUS_CONFIG_ROOT:-${HOME}/.libero-openwam-plus}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
