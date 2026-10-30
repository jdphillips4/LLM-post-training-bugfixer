import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from bugfix.scoring import reward_from_score, score_completion
from bugfix.tasks import load_tasks, read_jsonl
from bugfix.train_utils import LORA, load_model_and_tokenizer, make_config, merge_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--samples", help="samples file; keeps only tasks with 0 < pass rate < 1")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()

    tasks = {t["id"]: t for t in load_tasks(a.tasks, "train")}
    keep = None
    if a.samples:  # all-pass / all-fail groups give zero GRPO signal
        rates = defaultdict(list)
        for r in read_jsonl(a.samples):
            rates[r["task_id"]].append(r["passed"] and not r["regression"])
        keep = {t for t, v in rates.items() if 0 < sum(v) / len(v) < 1}
        print(f"{len(keep)} tasks with mixed outcomes")
    data = [{"prompt": p["prompt"], "task_id": p["task_id"]} for p in read_jsonl(a.prompts)
            if p["task_id"] in tasks and (keep is None or p["task_id"] in keep)]

    def test_reward(prompts, completions, task_id, **kwargs):
        def one(x):
            tid, c = x
            text = c[0]["content"] if isinstance(c, list) else c
            return reward_from_score(score_completion(tasks[tid], text))
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(one, zip(task_id, completions)))

    model, tok = load_model_and_tokenizer(a.base, a.four_bit)
    cfg = make_config(
        GRPOConfig, output_dir=f"{a.out}/adapter", learning_rate=a.lr,
        num_train_epochs=a.epochs, num_generations=a.generations,
        per_device_train_batch_size=a.generations, gradient_accumulation_steps=4,
        max_prompt_length=5120, max_completion_length=768, temperature=0.9,
        logging_steps=1, save_strategy="no", bf16=True, gradient_checkpointing=True,
        report_to="none")
    trainer = GRPOTrainer(model=model, reward_funcs=[test_reward], args=cfg,
                          train_dataset=Dataset.from_list(data),
                          processing_class=tok, peft_config=LORA)
    trainer.train()
    trainer.save_model(f"{a.out}/adapter")
    merge_lora(a.base, f"{a.out}/adapter", f"{a.out}/merged")


if __name__ == "__main__":
    main()