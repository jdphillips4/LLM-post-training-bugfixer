import argparse

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from bugfix.train_utils import LORA, load_model_and_tokenizer, make_config, merge_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()

    model, tok = load_model_and_tokenizer(a.base, a.four_bit)
    ds = load_dataset("json", data_files=a.data, split="train")  # prompt / completion columns
    cfg = make_config(
        SFTConfig, output_dir=f"{a.out}/adapter", num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="no", bf16=True, gradient_checkpointing=True,
        max_length=a.max_length, max_seq_length=a.max_length,   # name differs by TRL version
        completion_only_loss=True, report_to="none")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=LORA)
    trainer.train()
    trainer.save_model(f"{a.out}/adapter")
    merge_lora(a.base, f"{a.out}/adapter", f"{a.out}/merged")


if __name__ == "__main__":
    main()
    