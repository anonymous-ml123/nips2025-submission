# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.

the code modified from: https://github.com/vllm-project/vllm/blob/main/examples/vllm/vllm_model_example.py
"""

from typing import NamedTuple, Optional
from vllm import EngineArgs
from vllm.lora.request import LoRARequest


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# Idefics3-8B-Llama3
def run_idefics3(modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"
    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 3 * 364
            },
        },
        limit_mm_per_prompt={"image": 1},
    )
    prompts = [
        "<|begin_of_text|>User:<image>{}<end_of_utterance>\nAssistant:"]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# SmolVLM2-2.2B-Instruct
def run_smolvlm(modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"
    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        mm_processor_kwargs={
            "max_image_size": {
                "longest_edge": 384
            },
        },
        limit_mm_per_prompt={"image": 1},
    )
    prompts = ["<|im_start|>User:<image>{}<end_of_utterance>\nAssistant:"]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Kimi-VL
def run_kimi_vl(modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        "<|im_user|>user<|im_middle|><|media_start|>image<|media_content|>"
        "<|media_pad|><|media_end|>{}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"]

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-OneVision
def run_llava_onevision(modality: str,
                        model_path: str) -> ModelRequestData:
    assert modality == "image"
    
    prompts = [
        "<|im_start|>user <image>\n{}<|im_end|> \
    <|im_start|>assistant\n"]

    engine_args = EngineArgs(
        model=model_path,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )
    
    
def run_qwen2_5_vl(modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"
    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    placeholder = "<|image_pad|>"

    prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         "{}<|im_end|>\n"
         "<|im_start|>assistant\n"
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )



vllm_model_example_map = {
    "idefics3-8b-llama3": run_idefics3,
    "kimi-vl-3b": run_kimi_vl,
    "llava-onevision-7b": run_llava_onevision,
    "smolvlm-2b": run_smolvlm,
    'qwen2_5_vl_3b': run_qwen2_5_vl,
}