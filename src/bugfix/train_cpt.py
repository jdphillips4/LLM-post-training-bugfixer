"""Continued pretraining on a local corpus of Python files.
Do NOT include the repos you evaluate on. Use a low LR and re-check instruction following:
CPT on an instruct model can degrade it."""
import argparse
from pathlib import Path

from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from bugfix.train_utils import LORA, load_model_and_tokenizer, make_config, merge_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()

    model, tok = load_model_and_tokenizer(a.base, a.four_bit)
    ids = []
    for p in Path(a.corpus_dir).rglob("*.py"):
        ids += tok(p.read_text(errors="ignore"), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    blocks = [ids[i:i + a.block] for i in range(0, len(ids) - a.block, a.block)]
    ds = Dataset.from_dict({"input_ids": blocks})
    print(f"{len(ids)} tokens -> {len(blocks)} blocks")

    if a.four_bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()
    model = get_peft_model(model, LORA)
    args = make_config(TrainingArguments, output_dir=f"{a.out}/adapter", num_train_epochs=a.epochs,
                       per_device_train_batch_size=1, gradient_accumulation_steps=16,
                       learning_rate=a.lr, bf16=True, gradient_checkpointing=True,
                       logging_steps=10, save_strategy="no", report_to="none")
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(f"{a.out}/adapter")
    merge_lora(a.base, f"{a.out}/adapter", f"{a.out}/merged")


if __name__ == "__main__":
    main()