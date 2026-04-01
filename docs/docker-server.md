# CapsWriter Server Docker 部署

這份文件只覆蓋 **Server 端**。Client 仍維持原本 Windows 用法。

## 目標

- 以 Docker Image 方式封裝 CapsWriter-Offline Server
- 在 Linux / Docker headless 環境中可穩定啟動
- 模型檔與日誌走宿主機 volume，不把大模型烤進映像

## 內容物

- `docker/server/Dockerfile`：Server 專用 production image
- `docker-compose.yml`：本機啟動與模型準備
- `docker/server/download_models.py`：自動下載官方模型資產
- `docker/server/healthcheck.py`：容器健康檢查
- `requirements-server-docker.txt`：Linux/headless 最小依賴集

## 預設策略

- 預設模型：`qwen_asr`
- 預設模式：`GPU 優先（auto） + headless，失敗回退 CPU`
- Docker 內強制關閉 tray、DirectML；Vulkan / CPU 由 runtime 自動判斷
- 模型掛載到 `./models`，日誌持久化到 Docker named volume

> 目前公開 image 的優先目標是 **Tesla P4 / Pascal** 這類已驗證硬體。其他 GPU 世代的廣泛相容性仍列為後續 TODO。

這樣做是為了先得到最穩定、最容易在 Linux 上落地的 server 版本。

這個 image 會在**容器啟動時自動下載缺失模型**；若模型是 `qwen_asr` 或 `fun_asr_nano`，也會一併自動準備對應的 Linux `llama.cpp` 共享庫。`CAPSWRITER_INFERENCE_HARDWARE=auto` 時，容器會先偵測 GPU runtime：有可見 GPU 就優先用 Vulkan backend，沒有就自動回退 CPU backend。對 Pascal / P4，這個 image 也會優先嘗試把 ONNX encoder / CTC 放到 CUDA 上。

## 啟動前準備

```bash
cp docker/server/.env.example .env
```

根目錄 `.env` 只作為本機啟動設定使用，已被 Docker build context 排除，不會被打進 image。

## 1. 準備公開 image

預設情況下，這份 compose 會直接使用公開 image：

```text
ghcr.io/df-wu/capswriter-offline-server:latest
```

你可以在 `.env` 裡覆蓋成別的 tag 或私有 image，但大多數情況不需要。

## 2. 啟動 Server

```bash
docker compose up -d capswriter-server
```

預設會在容器啟動時自動下載 `qwen_asr` 所需模型，並以 `CAPSWRITER_INFERENCE_HARDWARE=auto` 進行 GPU 優先啟動。

如果你要切到 `fun_asr_nano`，先在 `.env` 裡設定：

```env
CAPSWRITER_MODEL_TYPE=fun_asr_nano
```

可選值：

- `fun_asr_nano`
- `qwen_asr`

目前這次交付只把 **預設 `qwen_asr`** 與 **ENV 切換的 `fun_asr_nano`** 當成正式支援目標。

第一次冷啟動可能需要比較久，因為容器會先下載模型與 `llama.cpp` 共享庫，再啟動服務。

### Qwen preset

`CAPSWRITER_QWEN_PRESET` 目前只保留兩個正式選項：

- `default`：一般使用的主力預設
- `low_vram_gpu`：保留 ONNX GPU，加上 CPU llama，降低顯存壓力

舊的 `balanced` / `quality` 目前只作為相容別名，會映射到 `default`。

### Inference hardware 策略

- `CAPSWRITER_INFERENCE_HARDWARE=auto`：預設值。容器內看得到 GPU runtime 時優先走 Vulkan，否則回退 CPU。
- `CAPSWRITER_INFERENCE_HARDWARE=gpu`：仍然會先嘗試 GPU；若 runtime 不可見，會回退 CPU，不讓服務直接失敗。
- `CAPSWRITER_INFERENCE_HARDWARE=cpu`：強制 CPU。

這份 compose 檔已經把 GPU request 定義在同一個檔案裡；`auto` 會在 GPU 可見時走 Vulkan，不可用時回退 CPU。

如果你要用同一份 compose 直接走 CPU-only 啟動，把 `CAPSWRITER_GPU_DEVICE_COUNT=0`；若要請求 GPU，維持預設 `all` 即可。

Compose 層額外提供 `CAPSWRITER_GPU_DEVICE_COUNT`：

- `all`：向容器請求 GPU（預設）
- `0`：不向容器請求 GPU，適合 CPU-only 啟動

不需要額外的 helper service。啟動 `capswriter-server` 時，容器會自動下載缺失模型與 backend。

## 3. 直接使用 image

如果你要直接使用 image，也可以：

```bash
docker run -d --name capswriter-server \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e CAPSWRITER_INFERENCE_HARDWARE=auto \
  -p 6016:6016 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/hot-server.txt:/app/hot-server.txt" \
  ghcr.io/df-wu/capswriter-offline-server:latest
```

改成 `fun_asr_nano`：

```bash
docker run -d --name capswriter-server \
  -e CAPSWRITER_MODEL_TYPE=fun_asr_nano \
  -e CAPSWRITER_INFERENCE_HARDWARE=auto \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -p 6016:6016 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/hot-server.txt:/app/hot-server.txt" \
  ghcr.io/df-wu/capswriter-offline-server:latest
```

## 4. 查看狀態

```bash
docker compose ps
docker compose logs -f capswriter-server
```

健康檢查通過後，Server 會在：

```text
ws://127.0.0.1:${CAPSWRITER_SERVER_PORT}
```

預設是：

```text
ws://127.0.0.1:6016
```

## 5. 停止

```bash
docker compose down
```

## Production 設計說明

### 1) 為什麼不用原本的 `requirements-server.txt`

原始清單帶有 Windows / UI 導向依賴，例如 DirectML 與 tray 相關套件。Docker 版改用 `requirements-server-docker.txt`，避免把 headless Linux 不需要的東西也裝進去。

### 2) 為什麼模型不打進 image

模型體積很大，而且不同部署節點可能選不同模型。把模型做成 volume 可以：

- 降低 image 體積
- 避免每次改 code 都重拉大模型
- 讓升級與回滾更可控

### 3) 為什麼強制關掉 tray / DirectML，但把 Vulkan 改成 runtime 自動判斷

- tray 在 headless Linux 容器沒有意義
- DirectML 是 Windows 路徑
- Vulkan 在容器裡是否可用，取決於 GPU 裝置與驅動是否真的被暴露給容器；因此 Docker 版會在啟動時先判斷，有 GPU 就優先用 Vulkan，沒有就回退 CPU

### 4) 為什麼日誌不用宿主機 bind mount

本機直接 bind `./logs` 很容易遇到 UID / GID 不一致，讓 non-root container 寫不進去。Docker 版改用 named volume 保留持久化，同時避免權限問題；即時排障則直接用 `docker compose logs`。

## 常用覆寫參數

可透過 `.env` 或 compose environment 覆寫：

- `CAPSWRITER_SERVER_IMAGE`
- `CAPSWRITER_MODEL_TYPE`
- `CAPSWRITER_QWEN_PRESET`
- `CAPSWRITER_INFERENCE_HARDWARE`
- `CAPSWRITER_SERVER_PORT`
- `CAPSWRITER_LOG_LEVEL`
- `CAPSWRITER_NUM_THREADS`
- `CAPSWRITER_REMOVE_MODEL_ARCHIVES`

程式也支援：

- `CAPSWRITER_ENABLE_TRAY`
- `CAPSWRITER_FUNASR_DML_ENABLE`
- `CAPSWRITER_FUNASR_VULKAN_ENABLE`
- `CAPSWRITER_QWEN_USE_DML`
- `CAPSWRITER_QWEN_VULKAN_ENABLE`

## 驗證標準

完成部署後，至少要看到：

1. `docker compose ps` 顯示 `healthy`
2. logs 出現模型載入完成與 WebSocket 監聽訊息
3. 本機可以連到對應 port

## TODO

- 驗證更新的 NVIDIA GPU 世代是否同樣適合這條 CUDA 11.8 / ORT 1.18 image 線
- 視驗證結果決定是否維持單一 public image，或改成多個硬體相容 tag
