# CapsWriter-Offline Linux Server Fork

Linux- and Docker-focused server fork of [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline).

This fork keeps the upstream recognition stack and deployment model where it makes sense, but repackages the **server side** for Linux hosts, containers, and GPU-enabled machines.

**Current release priority:** this fork is currently optimized first for **Tesla P4 / Pascal-class NVIDIA GPUs** running on Linux Docker. Wider GPU compatibility remains a follow-up task.

## Overview

The upstream project is a polished offline speech input tool built primarily for Windows desktop use. This fork exists for a different deployment target:

- run the CapsWriter server on Linux,
- package it as a Docker image,
- start it with predictable runtime behavior,
- prefer GPU acceleration when available, and
- fall back to CPU when GPU execution is not usable.

This repository is not a Linux desktop port of the full product. It is a **Linux server fork**.

## Why this fork exists

CapsWriter-Offline already solves the desktop-side problem well: low-latency offline speech input with strong model support and a practical user experience.

What it did not provide was a deployment path for users who want to run the recognition server on:

- a Linux workstation,
- a home server,
- a Docker host,
- or a GPU machine used as a shared inference box.

This fork closes that gap. Its goal is simple: make the server side easier to build, run, operate, and recover on Linux without changing the upstream project's identity.

## Scope

### What this fork does

- packages the server as a Linux Docker image,
- uses `qwen_asr` as the default model,
- supports `fun_asr_nano` through environment variables,
- downloads required model assets at container startup,
- prefers GPU inference when available,
- falls back to CPU when GPU inference is unavailable or fails backend probing,
- provides a single `docker-compose.yml` entry point.

### Current deployment target

At the moment, the most thoroughly validated GPU path is:

- Linux
- Docker
- NVIDIA Tesla P4 / Pascal
- `qwen_asr` and `fun_asr_nano`

The public image is published with that target in mind.

### What this fork does not do

- replace the upstream Windows client,
- turn the project into a native Linux desktop application,
- or redefine the upstream project's product direction.

If you want the original Windows experience, use the upstream repository. If you want a Linux-friendly server deployment path, use this fork.

## What changed from upstream

The main change is not model selection. The main change is **delivery and operation**.

### 1. Container-first server packaging

This fork adds:

- a Linux-oriented Docker image,
- a Docker Compose entry point,
- a container health check,
- an automatic model/bootstrap download flow,
- and a startup chain designed for long-running server use.

The goal is operational consistency: build, run, restart, inspect, and recover the server in a standard way.

### 2. Deployment-friendly configuration

The upstream project is naturally centered around local configuration files. This fork adds a deployment surface that works well with Docker and remote hosts, including:

- `CAPSWRITER_MODEL_TYPE`
- `CAPSWRITER_INFERENCE_HARDWARE`
- `CAPSWRITER_GPU_DEVICE_COUNT`
- `CAPSWRITER_SERVER_PORT`
- `CAPSWRITER_LOG_LEVEL`

This does not replace upstream configuration. It makes server deployment easier to automate.

### 3. GPU-first, not GPU-only

This fork treats hardware selection as a runtime decision:

- if a usable GPU runtime is visible, prefer Vulkan,
- if no GPU runtime is available, fall back to CPU,
- if a GPU backend is selected but fails probing, fall back to CPU before starting the server.

The target behavior is not “force GPU at all costs.” The target behavior is: **use GPU when it helps, but keep the service available when it does not.**

### 4. Linux and headless runtime fixes

This fork also addresses deployment-level issues that matter in containers:

- no blocking `input()` prompts in non-interactive runs,
- stable log path handling,
- a clean Docker build context,
- model and runtime assets downloaded on demand,
- one Compose file that works for both GPU-backed and CPU-only startup.

## Supported models

- `qwen_asr` — default
- `fun_asr_nano` — selected with `CAPSWRITER_MODEL_TYPE=fun_asr_nano`

## Inference hardware behavior

### Runtime selection

- `CAPSWRITER_INFERENCE_HARDWARE=auto` — default; prefer GPU, fall back to CPU
- `CAPSWRITER_INFERENCE_HARDWARE=gpu` — try GPU first; still fall back to CPU if probing fails
- `CAPSWRITER_INFERENCE_HARDWARE=cpu` — force CPU

### Compose-level GPU request

- `CAPSWRITER_GPU_DEVICE_COUNT=all` — request GPU devices (default)
- `CAPSWRITER_GPU_DEVICE_COUNT=0` — request no GPU devices; useful for CPU-only startup

## Quick start

### 1. Create a local `.env`

```bash
cp docker/server/.env.example .env
```

Default values:

```env
CAPSWRITER_MODEL_TYPE=qwen_asr
CAPSWRITER_INFERENCE_HARDWARE=auto
CAPSWRITER_GPU_DEVICE_COUNT=all
CAPSWRITER_SERVER_PORT=6016
```

The repository-root `.env` file is for local deployment only. It is excluded from the Docker build context and is not baked into the image.

### 2. Start the default server

```bash
docker compose up -d capswriter-server
```

### 3. Switch to `fun_asr_nano`

```bash
CAPSWRITER_MODEL_TYPE=fun_asr_nano \
CAPSWRITER_INFERENCE_HARDWARE=auto \
docker compose up -d --force-recreate capswriter-server
```

### 4. Start in CPU-only mode

```bash
CAPSWRITER_GPU_DEVICE_COUNT=0 \
CAPSWRITER_INFERENCE_HARDWARE=auto \
docker compose up -d --force-recreate capswriter-server
```

## Startup flow

At runtime, the container does the following:

1. reads model and hardware settings from the environment,
2. decides whether to use a Vulkan or CPU backend,
3. downloads missing model assets,
4. prepares Linux `llama.cpp` shared libraries for `qwen_asr` and `fun_asr_nano`,
5. probes the selected GPU backend when applicable,
6. falls back to CPU if the probe fails,
7. starts the server and exposes health status.

This order is intentional. It reduces “start and hope” behavior and makes failure handling more predictable.

There is no separate helper service in Compose. Starting `capswriter-server` is enough.

## Who this fork is for

This fork is a better fit than the upstream README if you want to:

- run CapsWriter server on Linux,
- operate it with Docker,
- centralize model bootstrap and backend preparation,
- use GPU when available without losing CPU fallback,
- keep the upstream recognition path while avoiding Linux deployment work from scratch.

## Relationship to the upstream project

This repository is explicitly based on the original project:

- Upstream: [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- Upstream focus: offline speech input on Windows
- This fork: Linux- and Docker-oriented server deployment

This fork extends the upstream server deployment story. It does not replace the upstream project.

## Project status

This fork is intentionally narrow in scope.

It is meant to make the **server** easier to deploy on Linux. It is not meant to become a general-purpose rewrite of the whole project.

## TODO

- validate the current GPU line on newer NVIDIA architectures beyond Pascal,
- decide whether a single GPU image line is sufficient or whether multiple runtime tags should be published,
- revisit broader Linux GPU compatibility after the Pascal/P4 path is stable in real use.

## Acknowledgements

Thanks to the upstream project and its author:

- [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)

This fork also depends on the work of the upstream ecosystem around it, including:

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

## See also

If you want the original project documentation, desktop workflow, and Windows-focused usage guide, start with the upstream repository:

- <https://github.com/HaujetZhao/CapsWriter-Offline>
