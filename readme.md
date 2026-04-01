# CapsWriter-Offline Linux Server Fork

這個 fork 保留了 CapsWriter-Offline 原本的辨識能力與 C/S 架構，但把重心放在 **Linux 上可長期運作的 server/container 落地**。如果你要的是 Windows 桌面端完整體驗，請看 upstream；如果你要的是可部署、可重啟、可掛 volume 的 Linux server，這份 README 就是最快的 onboarding 入口。

## 這個專案在做什麼

- **Client 端**：仍以原專案的 Windows 使用情境為主。
- **Server 端**：這個 fork 針對 Linux、Docker、headless、GPU/CPU fallback 做了完整整理。
- **容器化目標**：用一個 image + 一份 compose 啟動 server，缺的模型在容器啟動時自動下載，GPU 可用就優先使用，不可用就回退 CPU。

## 先理解整個專案，但實際先從 server 開始

核心啟動鏈如下：

1. [`docker-compose.yml`](docker-compose.yml) 提供預設服務、port、volume 與 environment。
2. [`docker/server/entrypoint.sh`](docker/server/entrypoint.sh) 判斷 GPU/CPU 路徑。
3. [`docker/server/download_models.py`](docker/server/download_models.py) 下載模型與 Linux `llama.cpp` 共享庫。
4. [`docker/server/probe_backend.py`](docker/server/probe_backend.py) 驗證 GPU backend 是否真的可用；失敗就回退 CPU。
5. [`start_server.py`](start_server.py) -> [`core_server.py`](core_server.py) 啟動 WebSocket server。
6. [`util/server/service.py`](util/server/service.py) 另外拉一個辨識子進程，避免模型推理阻塞主 WebSocket 流程。

如果你只想先用起來，記住三件事就夠了：

- **服務入口是 `docker compose up -d capswriter-server`**
- **設定主要靠 `.env`**
- **模型與熱詞都走宿主機 volume，不烤進 image**

## 專案內你最先會碰到的檔案

- [`readme.md`](readme.md)：快速 onboarding
- [`docker-compose.yml`](docker-compose.yml)：實際可直接跑的 compose
- [`.env.example`](.env.example)：建議複製成 `.env` 的範例設定
- [`docker-compose.example.yml`](docker-compose.example.yml)：給你改成自己部署版本的範例 compose
- [`hot-server.example.txt`](hot-server.example.txt)：server 熱詞範例檔
- [`config_server.py`](config_server.py)：server 端最終會讀到的設定與模型參數預設
- [`docs/docker-server.md`](docs/docker-server.md)：較完整的 Docker 設計說明

## 快速開始

### 前置需求

- Linux 主機
- Docker Engine
- Docker Compose Plugin
- 若要 GPU：NVIDIA driver + NVIDIA Container Toolkit

### 1. 準備本機設定檔

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
```

如果你只是先驗證流程，也可以直接使用 repo 既有的 [`hot-server.txt`](hot-server.txt)。

### 2. 直接用內建 compose 啟動

```bash
docker compose up -d capswriter-server
```

預設行為：

- model：`qwen_asr`
- hardware：`auto`（優先 GPU，不行就回退 CPU）
- port：`6016`
- models：掛到 `./models`
- logs：寫到 named volume `capswriter-server-logs`

### 3. 查看狀態

```bash
docker compose ps
docker compose logs -f capswriter-server
```

健康檢查通過後，預設 WebSocket 位址是：

```text
ws://127.0.0.1:6016
```

### 4. 停止

```bash
docker compose down
```

## 最小可用設定

這是最常用、最值得先懂的幾個環境變數：

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `CAPSWRITER_SERVER_IMAGE` | `ghcr.io/df-wu/capswriter-offline-server:latest` | 要拉的 image |
| `CAPSWRITER_MODEL_TYPE` | `qwen_asr` | `qwen_asr` / `fun_asr_nano` |
| `CAPSWRITER_QWEN_PRESET` | `default` | `default` / `low_vram_gpu` |
| `CAPSWRITER_INFERENCE_HARDWARE` | `auto` | `auto` / `gpu` / `cpu` |
| `CAPSWRITER_GPU_DEVICE_COUNT` | `all` | `all` / `0` |
| `CAPSWRITER_SERVER_PORT` | `6016` | 對外 WebSocket port |
| `CAPSWRITER_LOG_LEVEL` | `INFO` | 日誌等級 |
| `CAPSWRITER_NUM_THREADS` | `4` | CPU 階段執行緒數 |

## 範例設定檔

repo root 已提供可直接複製的範例：

- [`.env.example`](.env.example)
- [`hot-server.example.txt`](hot-server.example.txt)
- [`docker-compose.example.yml`](docker-compose.example.yml)

建議做法：

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
cp docker-compose.example.yml docker-compose.local.yml
docker compose -f docker-compose.local.yml up -d capswriter-server
```

如果你不需要自訂 compose，就直接使用 repo 內建的 [`docker-compose.yml`](docker-compose.yml)。

## 常見啟動情境

### 預設 `qwen_asr`

```bash
docker compose up -d capswriter-server
```

### 切到 `fun_asr_nano`

```bash
CAPSWRITER_MODEL_TYPE=fun_asr_nano \
docker compose up -d --force-recreate capswriter-server
```

### 強制 CPU-only

```bash
CAPSWRITER_GPU_DEVICE_COUNT=0 \
CAPSWRITER_INFERENCE_HARDWARE=cpu \
docker compose up -d --force-recreate capswriter-server
```

## 容器啟動時實際會發生什麼

這份 fork 的關鍵不是「只是把 Python 包進 Docker」，而是把 server 端整條 runtime 整理成可重複啟動的流程：

1. Compose 把 `.env` 與 volume 掛進容器。
2. entrypoint 依 `CAPSWRITER_INFERENCE_HARDWARE`、GPU 裝置可見性與 preset 決定 backend。
3. 如果模型不存在，先自動下載到 `./models`。
4. 若模型是 `qwen_asr` 或 `fun_asr_nano`，也會補齊 Linux `llama.cpp` 共享庫。
5. 若走 GPU 路徑，先 probe backend；probe 失敗就降回 CPU ONNX。
6. 最後才啟動 WebSocket server，並在另一個子進程載入辨識模型。

這樣做的好處是：

- 第一次冷啟動比較慢，但之後重啟行為可預期
- GPU 不可用時服務不會直接整個失敗
- 模型、熱詞、日誌都能跟容器生命週期分離

## Volume 與持久化

[`docker-compose.yml`](docker-compose.yml) 目前會掛這些路徑：

- `./models:/app/models`
- `./hot-server.txt:/app/hot-server.txt`
- `capswriter-server-logs:/app/logs`

含義如下：

- `models/`：存模型、下載快取與共享庫準備結果
- `hot-server.txt`：server 端熱詞檔，主要給 `fun_asr_nano` 的 decoder 參考
- `capswriter-server-logs`：容器內 server log 持久化

## 驗證你真的啟動成功了

至少確認這三件事：

1. `docker compose ps` 顯示 `healthy`
2. `docker compose logs -f capswriter-server` 能看到模型載入完成與監聽訊息
3. client 端或測試程式可以連到 `ws://127.0.0.1:${CAPSWRITER_SERVER_PORT}`

## 這份 fork 與 upstream 的關係

- upstream 專案：<https://github.com/HaujetZhao/CapsWriter-Offline>
- upstream 強項：Windows 桌面端語音輸入體驗
- 這個 fork：Linux / Docker / headless server 部署

這不是整個產品的 Linux 桌面移植版；這是 **server 端部署路線** 的整理與落地。

## 深入閱讀

- [`docs/docker-server.md`](docs/docker-server.md)：Docker 版設計與部署說明
- [`config_server.py`](config_server.py)：環境變數如何落到 server 端設定
- [`docker/server/Dockerfile`](docker/server/Dockerfile)：image 組成
- [`docker/server/.env.example`](docker/server/.env.example)：原始 Docker 版 env 範例來源

## Acknowledgements

- Upstream: [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
