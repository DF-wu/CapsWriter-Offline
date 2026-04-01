# coding=utf-8
import os
import time
import re
import codecs
import dataclasses
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from typing import Optional, List

from .schema import (
    MsgType,
    StreamingMessage,
    DecodeResult,
    ASREngineConfig,
    TranscribeResult,
    ForcedAlignItem,
    ForcedAlignResult,
)
from .utils import normalize_language_name, validate_language
from .encoder import QwenAudioEncoder
from . import llama


@dataclasses.dataclass
class ASRS_Segment:
    """管理分片记忆及其物理时间坐标"""

    idx: int
    audio_start: float
    audio_end: float
    text: str = ""
    items: List[ForcedAlignItem] = None


class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端) - 统一辅助进程架构"""

    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose:
            print(
                f"--- [QwenASR] 初始化引擎 (DML: {config.use_dml}, Vulkan: {config.vulkan_enable}) ---"
            )

        # 设置图形加速环境
        if not config.vulkan_enable:
            os.environ["VK_ICD_FILENAMES"] = "none"  # 禁止 Vulkan
        if config.vulkan_force_fp32:
            os.environ["GGML_VK_DISABLE_F16"] = (
                "1"  # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）
            )

        self.llama_mod = llama  # keep reference

        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)

        # 1. 初始化 Encoder
        self.encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            use_cuda=config.use_cuda,
            use_dml=config.use_dml,
            pad_to=config.pad_to,
            verbose=self.verbose,
        )

        # 2. 初始化 Aligner (可选)
        self.aligner = None
        if config.enable_aligner and config.align_config:
            from .aligner import QwenForcedAligner

            self.aligner = QwenForcedAligner(config.align_config)

        # 3. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(
            self.model,
            n_ctx=config.n_ctx,
            n_batch=config.llama_n_batch,
            n_ubatch=config.llama_n_ubatch,
            embeddings=False,
            flash_attn=config.llama_flash_attn,
            offload_kqv=config.llama_offload_kqv,
        )

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

        self._token_cache = {}
        self._prefix_embd_cache = {}
        self._suffix_head_embd_cache = {}

    def shutdown(self):
        if self.verbose:
            print("--- [QwenASR] 引擎已关闭 ---")

    def _build_prompt_embd(
        self,
        audio_embd: np.ndarray,
        prefix_text: str,
        context: Optional[str],
        language: Optional[str],
    ):
        def tk_cached(text: str) -> list[int]:
            cached = self._token_cache.get(text)
            if cached is None:
                cached = self.model.tokenize(text)
                self._token_cache[text] = cached
            return cached

        t_prompt_start = time.perf_counter()

        context_key = context or "You are a helpful assistant."
        prefix_embd = self._prefix_embd_cache.get(context_key)
        t_static_start = time.perf_counter()
        if prefix_embd is None:
            prefix_tokens = (
                [self.ID_IM_START]
                + tk_cached(f"system\n{context_key}")
                + [self.ID_IM_END]
                + [self.ID_IM_START]
                + tk_cached("user\n")
                + [self.ID_AUDIO_START]
            )
            prefix_embd = self.embedding_table[prefix_tokens]
            self._prefix_embd_cache[context_key] = prefix_embd

        suffix_key = language or ""
        suffix_head_embd = self._suffix_head_embd_cache.get(suffix_key)
        if suffix_head_embd is None:
            suffix_head = "assistant\n"
            if language:
                suffix_head += f"language {language}"
            suffix_head_tokens = (
                [self.ID_AUDIO_END]
                + [self.ID_IM_END]
                + [self.ID_IM_START]
                + tk_cached(suffix_head)
                + [self.ID_ASR_TEXT]
            )
            suffix_head_embd = self.embedding_table[suffix_head_tokens]
            self._suffix_head_embd_cache[suffix_key] = suffix_head_embd
        prompt_static_time = time.perf_counter() - t_static_start

        t_history_tokenize_start = time.perf_counter()
        history_tokens = tk_cached(prefix_text) if prefix_text else []
        prompt_history_tokenize_time = time.perf_counter() - t_history_tokenize_start

        t_history_embed_start = time.perf_counter()
        if history_tokens:
            history_embd = self.embedding_table[history_tokens]
        else:
            history_embd = np.empty((0, self.model.n_embd), dtype=np.float32)
        prompt_history_embed_time = time.perf_counter() - t_history_embed_start

        t_concat_start = time.perf_counter()
        total_embd = np.concatenate(
            [
                prefix_embd,
                audio_embd.astype(np.float32, copy=False),
                suffix_head_embd,
                history_embd,
            ],
            axis=0,
        )
        prompt_concat_time = time.perf_counter() - t_concat_start

        prompt_metrics = {
            "prompt_build_time": time.perf_counter() - t_prompt_start,
            "prompt_static_time": prompt_static_time,
            "prompt_history_tokenize_time": prompt_history_tokenize_time,
            "prompt_history_embed_time": prompt_history_embed_time,
            "prompt_concat_time": prompt_concat_time,
            "prompt_prefix_tokens": int(prefix_embd.shape[0]),
            "prompt_suffix_tokens": int(suffix_head_embd.shape[0]),
            "prompt_history_tokens": len(history_tokens),
        }

        return total_embd, prompt_metrics

    def _decode(
        self,
        full_embd: np.ndarray,
        prefix_text: str,
        rollback_num: int,
        is_last_chunk: bool = False,
        temperature: float = 0.4,
        streaming: bool = True,
    ) -> DecodeResult:
        """底层方法：执行单次 LLM 生成循环（物理推理）"""
        result = DecodeResult()

        total_len = full_embd.shape[0]
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate(
            [pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)]
        )
        batch = self.llama_mod.LlamaBatch(
            max(total_len * 4, 8192), self.model.n_embd, 1
        )
        batch.set_embd(full_embd, pos=pos_arr)

        # 1. Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start

        # 2. Generation Loop（使用新采样器和随机种子）
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        # 每次解码使用新的随机种子
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = self.llama_mod.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(512):  # Max new tokens per chunk
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break

            if self.ctx.decode_token(last_sampled_token) != 0:
                break

            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    if streaming:
                        print(
                            re.sub(r"([，。？！：,\.])", r"\1\n", piece),
                            end="",
                            flush=True,
                        )
                    stable_text_acc += piece

            # 熔断检查：检测重复循环
            if len(stable_tokens) > 15:
                if len(set(stable_tokens[-15:])) <= 3:
                    result.is_aborted = True
                    break

            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1

        gen_time = time.time() - t_gen_start
        del sampler  # 释放采样器资源
        del batch

        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    if streaming:
                        print(
                            re.sub(r"([，。？！：,\.])", r"\1\n", piece),
                            end="",
                            flush=True,
                        )
                    stable_text_acc += piece
            final_p = text_decoder.decode(b"", final=True)
            if final_p:
                print(final_p, end="", flush=True)
                stable_text_acc += final_p

        # 填充结果（内核输出标准化）
        result.text = stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        result.n_generate = n_gen_tokens
        return result

    def _safe_decode(
        self,
        full_embd: np.ndarray,
        prefix_text: str,
        rollback_num: int,
        is_last_chunk: bool,
        temperature: float,
        streaming: bool = True,
    ) -> DecodeResult:
        """带熔断加温重试的高层推理封装"""
        for i in range(4):
            res = self._decode(
                full_embd,
                prefix_text,
                rollback_num,
                is_last_chunk,
                temperature,
                streaming=streaming,
            )
            if not res.is_aborted:
                break
            temperature += 0.3
            res.text += "====解码有误，强制熔断===="
            print(f"\n\n[!] 触发重试 (Temp -> {temperature:.1f})\n")
        return res

    def _print_stats(self, stats: dict, audio_duration: float, t_total: float):
        """打印转录过程的性能统计指标"""
        rtf = t_total / audio_duration if audio_duration > 0 else 0
        pre_speed = (
            stats["prefill_tokens"] / stats["prefill_time"]
            if stats["prefill_time"] > 0
            else 0
        )
        gen_speed = (
            stats["decode_tokens"] / stats["decode_time"]
            if stats["decode_time"] > 0
            else 0
        )

        print(f"\n\n📊 性能统计:")
        print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
        print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
        print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
        if stats.get("align_time"):
            print(f"  🔹 对齐耗时    : {stats['align_time']:.3f} 秒")
        print(f"  🔹 编码耗时    : {stats['encode_time']:.3f} 秒")
        print(
            f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)"
        )
        print(
            f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)"
        )

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        start_second: float = 0.0,
        duration: float = 0.0,
        temperature: float = 0.4,
        rollback_num: int = 5,
    ) -> TranscribeResult:
        """运行完整转录流水线 (从文件加载音频)"""
        from .utils import load_audio

        audio = load_audio(audio_file, start_second=start_second, duration=duration)

        return self.asr(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num,
        )

    def asr(
        self,
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 2,
        temperature: float = 0.4,
        rollback_num: int = 5,
    ) -> TranscribeResult:
        """运行完整转录流水线 (三级流水线：i+1 预取, i 识别, i-1 对齐)"""
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / sr

        # 记忆管理 (预定义所有分片的物理边界)
        all_segments: List[ASRS_Segment] = [
            ASRS_Segment(
                idx=i,
                audio_start=i * chunk_size_sec,
                audio_end=min((i + 1) * chunk_size_sec, total_duration),
            )
            for i in range(num_chunks)
        ]
        asr_memory = deque(maxlen=memory_chunks)  # 存储 (embd, text)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []

        # 统计指标
        stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "encode_time": 0.0,
            "align_time": 0.0,
            "prompt_build_time": 0.0,
            "prompt_static_time": 0.0,
            "prompt_history_tokenize_time": 0.0,
            "prompt_history_embed_time": 0.0,
            "prompt_concat_time": 0.0,
            "prompt_prefix_tokens": 0,
            "prompt_suffix_tokens": 0,
            "prompt_history_tokens": 0,
        }
        t_main_start = time.time()

        # --- 顺序同步处理循环 ---
        for i in range(num_chunks):
            # 1. 编码第 i 片段
            s, e = i * samples_per_chunk, min((i + 1) * samples_per_chunk, total_len)
            chunk_data = audio[s:e]
            if len(chunk_data) < samples_per_chunk:
                chunk_data = np.pad(
                    chunk_data, (0, samples_per_chunk - len(chunk_data))
                )

            audio_feature, enc_time = self.encoder.encode(chunk_data)
            stats["encode_time"] += enc_time
            was_last = i == num_chunks - 1

            # 2. 识别第 i 片段文字
            prefix_text = "".join([m[1] for m in asr_memory])
            combined_audio = np.concatenate(
                [m[0] for m in asr_memory] + [audio_feature], axis=0
            )
            full_embd, prompt_metrics = self._build_prompt_embd(
                combined_audio, prefix_text, context, language
            )
            stats["prompt_build_time"] += prompt_metrics["prompt_build_time"]
            stats["prompt_static_time"] += prompt_metrics["prompt_static_time"]
            stats["prompt_history_tokenize_time"] += prompt_metrics[
                "prompt_history_tokenize_time"
            ]
            stats["prompt_history_embed_time"] += prompt_metrics[
                "prompt_history_embed_time"
            ]
            stats["prompt_concat_time"] += prompt_metrics["prompt_concat_time"]
            stats["prompt_prefix_tokens"] += prompt_metrics["prompt_prefix_tokens"]
            stats["prompt_suffix_tokens"] += prompt_metrics["prompt_suffix_tokens"]
            stats["prompt_history_tokens"] += prompt_metrics["prompt_history_tokens"]

            # 带熔断加温重试的解码调用
            res = self._safe_decode(
                full_embd, prefix_text, rollback_num, was_last, temperature
            )

            # 更新记忆与统计
            all_segments[i].text = res.text
            asr_memory.append((audio_feature, res.text))

            total_full_text += res.text
            stats["prefill_tokens"] += res.n_prefill
            stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate
            stats["decode_time"] += res.t_generate

            # 3. 对齐第 i 片段 (同步)
            if self.aligner and res.text.strip():
                t_align_start = time.time()
                # 计算偏移（同步版本逻辑简化：直接使用片起点，不考虑前片动态边界）
                offset_sec = all_segments[i].audio_start
                s_smpl, e_smpl = (
                    int(offset_sec * sr),
                    int(all_segments[i].audio_end * sr),
                )
                audio_slice = audio[s_smpl:e_smpl]

                align_res = self.aligner.align(
                    audio_slice,
                    res.text,
                    language=language,
                    offset_sec=float(offset_sec),
                )
                all_segments[i].items = align_res.items
                all_aligned_items.extend(align_res.items)
                stats["align_time"] += time.time() - t_align_start

        # 4. 结果整理
        all_aligned_items.sort(key=lambda x: x.start_time)
        t_total = time.time() - t_main_start
        if self.verbose:
            self._print_stats(stats, total_duration, t_total)

        return TranscribeResult(
            text=total_full_text,
            alignment=ForcedAlignResult(items=all_aligned_items)
            if all_aligned_items
            else None,
            performance=stats,
        )
