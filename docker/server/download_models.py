import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from config_server import ModelPaths, ServerConfig


class Asset:
    def __init__(
        self,
        name: str,
        url: str,
        sha256: str,
        target_dir: Path,
        required_files: List[Path],
    ):
        self.name = name
        self.url = url
        self.sha256 = sha256
        self.target_dir = target_dir
        self.required_files = required_files

    @property
    def archive_path(self) -> Path:
        return Path("models") / ".downloads" / self.name

    def is_ready(self) -> bool:
        return all(path.exists() for path in self.required_files)


ASSETS = {
    "sensevoice": [
        Asset(
            name="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
            sha256="eece077adcd80eb90fc7905c3b11b9bab8588d810bde88ad694b8f3c78716320",
            target_dir=Path("models") / "SenseVoice-Small",
            required_files=[ModelPaths.sensevoice_model, ModelPaths.sensevoice_tokens],
        )
    ],
    "paraformer": [
        Asset(
            name="speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx.zip",
            sha256="49ce9dabea9812fd9cf510316ce2041fa45c6a5e361ee19025b8ef53d1c7af88",
            target_dir=Path("models") / "Paraformer",
            required_files=[ModelPaths.paraformer_model, ModelPaths.paraformer_tokens],
        ),
        Asset(
            name="sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.zip",
            sha256="082c932ecac8e31981b0a3a168ef994373012bd17dc1b6681b094f167c5a45bc",
            target_dir=Path("models") / "Punct-CT-Transformer",
            required_files=[ModelPaths.punc_model_dir],
        ),
    ],
    "fun_asr_nano": [
        Asset(
            name="Fun-ASR-Nano-GGUF.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Fun-ASR-Nano-GGUF.zip",
            sha256="3ca993df54c6f9eacf23e7c3a1dccc8121e506aaff787726648f604603f6af19",
            target_dir=Path("models") / "Fun-ASR-Nano",
            required_files=[
                ModelPaths.fun_asr_nano_gguf_encoder_adaptor,
                ModelPaths.fun_asr_nano_gguf_ctc,
                ModelPaths.fun_asr_nano_gguf_llm_decode,
                ModelPaths.fun_asr_nano_gguf_token,
            ],
        )
    ],
    "qwen_asr": [
        Asset(
            name="Qwen3-ASR-1.7B.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B.zip",
            sha256="897c66051a155a324c31684b768fe2e09b767372198c015c629a3b9088c1cf93",
            target_dir=Path("models") / "Qwen3-ASR",
            required_files=[
                ModelPaths.qwen3_asr_gguf_encoder_frontend,
                ModelPaths.qwen3_asr_gguf_encoder_backend,
                ModelPaths.qwen3_asr_gguf_llm_decode,
            ],
        )
    ],
}

LLAMA_CPP_ASSETS = {
    "cpu": Asset(
        name="llama-b7798-bin-ubuntu-x64.tar.gz",
        url="https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-ubuntu-x64.tar.gz",
        sha256="13e0fbbb2e1a3379ec5ed84ee9fe230f7d0c1dedbfc1b769424f0adc37f616b8",
        target_dir=Path("util") / "llama" / "bin",
        required_files=[],
    ),
    "vulkan": Asset(
        name="llama-b7798-bin-ubuntu-vulkan-x64.tar.gz",
        url="https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="5a4ee2db7e6f9d5c0a04741f79437b433e63862d72a962466b9f28f09f7ffff9",
        target_dir=Path("util") / "llama" / "bin",
        required_files=[],
    ),
}

LLAMA_TARGET_DIRS = [
    Path("util") / "llama" / "bin",
    Path("util") / "qwen_asr_gguf" / "inference" / "bin",
    Path("util") / "fun_asr_gguf" / "inference" / "bin",
]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _report(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(blocks * block_size, total_size)
        percent = downloaded * 100 / total_size
        print(f"\r下载中 {destination.name}: {percent:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, _report)
    print()


def _extract(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(target_dir)


def _llama_binaries_ready() -> bool:
    required_names = ["libggml.so", "libggml-base.so", "libllama.so", "libggml-cpu.so"]
    preferred_backend = os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
    if preferred_backend == "vulkan":
        required_names.append("libggml-vulkan.so")

    for target_dir in LLAMA_TARGET_DIRS:
        for required_name in required_names:
            if not (target_dir / required_name).exists():
                return False
    return True


def _extract_llama_binaries(archive: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with tarfile.open(archive, "r:gz") as tar_file:
            tar_file.extractall(temp_dir)

        shared_libraries = [
            file_path
            for file_path in temp_dir.rglob("*")
            if file_path.is_file() and ".so" in file_path.name
        ]

        if not shared_libraries:
            raise FileNotFoundError("未在 llama.cpp 压缩包中找到 Linux 共享库")

        for target_dir in LLAMA_TARGET_DIRS:
            target_dir.mkdir(parents=True, exist_ok=True)
            for library_path in shared_libraries:
                shutil.copy2(library_path, target_dir / library_path.name)

            cpu_backend = target_dir / "libggml-cpu.so"
            if not cpu_backend.exists():
                preferred_backend = None
                for candidate_name in [
                    "libggml-cpu-x64.so",
                    "libggml-cpu-haswell.so",
                    "libggml-cpu-sse42.so",
                ]:
                    candidate_path = target_dir / candidate_name
                    if candidate_path.exists():
                        preferred_backend = candidate_path
                        break

                if preferred_backend is not None:
                    shutil.copy2(preferred_backend, cpu_backend)


def _verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _prepare_llama_binaries() -> int:
    backend = os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
    if backend not in LLAMA_CPP_ASSETS:
        print(f"不支持的 llama.cpp backend: {backend}", file=sys.stderr)
        return 1

    if _llama_binaries_ready():
        print(f"llama.cpp Linux 共享库已就绪，backend={backend}，跳过下载")
        return 0

    asset = LLAMA_CPP_ASSETS[backend]
    archive_path = asset.archive_path
    if archive_path.exists():
        print(f"发现已下载 llama.cpp 压缩包，直接解压: {archive_path}")
    else:
        print(f"开始下载 {asset.name}")
        _download(asset.url, archive_path)

    if not _verify_sha256(archive_path, asset.sha256):
        print(f"llama.cpp 压缩包校验失败: {asset.name}", file=sys.stderr)
        return 1

    try:
        _extract_llama_binaries(archive_path)
    except Exception as error:
        print(f"解压 llama.cpp Linux 共享库失败: {error}", file=sys.stderr)
        return 1

    if not _llama_binaries_ready():
        print("llama.cpp Linux 共享库解压后仍不完整", file=sys.stderr)
        return 1

    print(f"llama.cpp Linux 共享库已准备完成，backend={backend}")
    return 0


def main() -> int:
    model_type = ServerConfig.model_type.lower()
    assets = ASSETS.get(model_type)
    if not assets:
        print(f"不支持的模型类型: {ServerConfig.model_type}", file=sys.stderr)
        return 1

    for asset in assets:
        if asset.is_ready():
            print(f"模型已就绪，跳过下载: {asset.name}")
            continue

        if asset.archive_path.exists():
            print(f"发现已下载压缩包，直接解压: {asset.archive_path}")
        else:
            print(f"开始下载 {asset.name}")
            _download(asset.url, asset.archive_path)

        if not _verify_sha256(asset.archive_path, asset.sha256):
            print(f"模型压缩包校验失败: {asset.name}", file=sys.stderr)
            return 1

        print(f"开始解压 {asset.name} -> {asset.target_dir}")
        _extract(asset.archive_path, asset.target_dir)

        if not asset.is_ready():
            print(f"解压完成后仍缺少模型文件: {asset.name}", file=sys.stderr)
            return 1

        print(f"模型已准备完成: {asset.name}")

    if model_type in {"qwen_asr", "fun_asr_nano"}:
        result = _prepare_llama_binaries()
        if result != 0:
            return result

    if os.getenv("CAPSWRITER_REMOVE_MODEL_ARCHIVES", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        downloads_dir = Path("models") / ".downloads"
        if downloads_dir.exists():
            shutil.rmtree(downloads_dir)
            print("已清理模型压缩包缓存")

    return 0


if __name__ == "__main__":
    sys.exit(main())
