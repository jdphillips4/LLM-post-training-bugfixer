import dataclasses

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")


def make_config(cls, **kw):
    """Build a TRL/HF config, dropping args this library version doesn't have."""
    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = [k for k in kw if k not in valid]
    if dropped:
        print(f"[warn] {cls.__name__}: ignoring unsupported args {dropped}")
    return cls(**{k: v for k, v in kw.items() if k in valid})


def load_model_and_tokenizer(name: str, four_bit: bool = False):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(name, **kwargs), tok


def merge_lora(base: str, adapter_dir: str, out_dir: str):
    """Merge into a bf16 copy of the base so the next stage / vLLM can load a plain model."""
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    print(f"merged model -> {out_dir}")