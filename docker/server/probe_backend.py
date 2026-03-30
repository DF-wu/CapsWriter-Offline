import sys
from typing import Any
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from config_server import FunASRNanoGGUFArgs, Qwen3ASRGGUFArgs, ServerConfig
from util.fun_asr_gguf import create_asr_engine as create_fun_asr_engine
from util.qwen_asr_gguf import create_asr_engine as create_qwen_asr_engine


def _cleanup(engine: Any) -> None:
    for method_name in ("cleanup", "shutdown"):
        method = getattr(engine, method_name, None)
        if callable(method):
            method()
            return


def main() -> int:
    model_type = ServerConfig.model_type.lower()
    engine = None

    try:
        if model_type == "qwen_asr":
            engine = create_qwen_asr_engine(
                **{
                    key: value
                    for key, value in Qwen3ASRGGUFArgs.__dict__.items()
                    if not key.startswith("_")
                }
            )
        elif model_type == "fun_asr_nano":
            engine = create_fun_asr_engine(
                **{
                    key: value
                    for key, value in FunASRNanoGGUFArgs.__dict__.items()
                    if not key.startswith("_")
                }
            )
        else:
            return 0
    except Exception as error:
        print(f"[capswriter] GPU backend probe failed: {error}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            _cleanup(engine)

    print(f"[capswriter] GPU backend probe passed for {model_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
