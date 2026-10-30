import argparse

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

from bugfix.train_utils import LORA, load_model_and_tokenizer, make_config, merge_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged SFT checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()

    model, tok = load_model_and_tokenizer(a.base, a.four_bit)
    ds = load_dataset("json", data_files=a.data, split="train")  # prompt / chosen / rejected
    cfg = make_config(
        DPOConfig, output_dir=f"{a.out}/adapter", beta=a.beta, learning_rate=a.lr,
        num_train_epochs=a.epochs, per_device_train_batch_size=1,
        gradient_accumulation_steps=16, logging_steps=5, save_strategy="no",
        bf16=True, gradient_checkpointing=True, report_to="none",
        max_length=6144, max_prompt_length=5120, max_completion_length=1024)
    # No ref_model: with a LoRA adapter, TRL uses the adapter-disabled model as reference.
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=LORA)
    trainer.train()
    trainer.save_model(f"{a.out}/adapter")
    merge_lora(a.base, f"{a.out}/adapter", f"{a.out}/merged")


if __name__ == "__main__":
    main()