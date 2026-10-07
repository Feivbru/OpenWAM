#!/usr/bin/env bash
# Worker / client env for OpenWAM RoboCasa GR1 MuJoCo EGL evaluation.
#
# Mirrors benchmarks/libero-plus/openwam_libero_plus_env.sh:
#   1) NVIDIA user-space EGL via VulkanDrive (no system graphic driver)
#   2) Prefer MagickWand lib_nogl so Mesa EGL cannot shadow NVIDIA
#
# Usage:
#   source benchmarks/robocasa_gr1/openwam_robocasa_gr1_env.sh
#   # then run_smoke.sh / single_eval.sh

_OW_GR1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_OW_ROOT="$(cd "${_OW_GR1_DIR}/../.." && pwd)"
_PI_ROOT="$(cd "${_OW_ROOT}/.." && pwd)"

# Prefer dedicated robocasa-gr1 conda python on PATH.
if [[ -x "/data/anaconda3/envs/robocasa-gr1/bin/python" ]]; then
  export PATH="/data/anaconda3/envs/robocasa-gr1/bin:${PATH}"
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
elif [[ -d "${MAGICK_LIB}" ]]; then
  echo "[openwam_robocasa_gr1_env] WARN: ${MAGICK_LIB_NOGL} missing; Magick GL may shadow NVIDIA EGL" >&2
  export LD_LIBRARY_PATH="${NVIDIA_VULKAN_LIB_DIR:-$HOME/.local/nvidia/lib64_vulkan}:${MAGICK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export ROBOCASA_GR1_PATH="${ROBOCASA_GR1_PATH:-${_OW_ROOT}/third_party/robocasa-gr1-tabletop-tasks}"
export ROBOCASA_GR1_PYTHON="${ROBOCASA_GR1_PYTHON:-/data/anaconda3/envs/robocasa-gr1/bin/python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
