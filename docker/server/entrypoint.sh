#!/bin/sh
set -eu

configure_backend() {
  inference_hardware="${CAPSWRITER_INFERENCE_HARDWARE:-${CAPSWRITER_GPU_MODE:-auto}}"
  llama_backend="cpu"

  gpu_visible="false"
  nvidia_visible="false"
  if [ -e /dev/nvidiactl ] || [ -d /dev/dri ]; then
    gpu_visible="true"
  fi
  if [ -e /dev/nvidiactl ]; then
    nvidia_visible="true"
  fi

  if [ "$inference_hardware" = "cpu" ]; then
    llama_backend="cpu"
  elif [ "$gpu_visible" = "true" ]; then
    llama_backend="vulkan"
  else
    llama_backend="cpu"
  fi

  if [ "$llama_backend" = "vulkan" ]; then
    export CAPSWRITER_LLAMA_BACKEND="vulkan"
    export CAPSWRITER_QWEN_VULKAN_ENABLE="true"
    export CAPSWRITER_FUNASR_VULKAN_ENABLE="true"
    if [ "$nvidia_visible" = "true" ]; then
      export CAPSWRITER_QWEN_USE_CUDA="true"
      export CAPSWRITER_FUNASR_USE_CUDA="true"
    else
      export CAPSWRITER_QWEN_USE_CUDA="false"
      export CAPSWRITER_FUNASR_USE_CUDA="false"
    fi
    echo "[capswriter] GPU runtime detected, preferring Vulkan backend"
  else
    export CAPSWRITER_LLAMA_BACKEND="cpu"
    export CAPSWRITER_QWEN_VULKAN_ENABLE="false"
    export CAPSWRITER_FUNASR_VULKAN_ENABLE="false"
    export CAPSWRITER_QWEN_USE_CUDA="false"
    export CAPSWRITER_FUNASR_USE_CUDA="false"
    if [ "$inference_hardware" = "gpu" ] || [ "$inference_hardware" = "auto" ]; then
      echo "[capswriter] GPU runtime unavailable, falling back to CPU backend"
    else
      echo "[capswriter] CPU mode forced"
    fi
  fi
}

fallback_to_cpu() {
  export CAPSWRITER_QWEN_USE_CUDA="false"
  export CAPSWRITER_FUNASR_USE_CUDA="false"
  echo "[capswriter] ONNX GPU probe failed, retrying with CPU ONNX while keeping llama backend"
}

configure_backend

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

python /app/docker/server/download_models.py

if [ "${CAPSWRITER_LLAMA_BACKEND:-cpu}" = "vulkan" ]; then
  if ! python /app/docker/server/probe_backend.py; then
    fallback_to_cpu
    python /app/docker/server/download_models.py
  fi
fi

exec python /app/start_server.py
