import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "arnir0/Tiny-LLM"


def resolve_dtype(dtype_name):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(model_id=MODEL_ID, dtype_name="float32"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolve_dtype(dtype_name),
    )
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
