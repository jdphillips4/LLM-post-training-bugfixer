import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bugfix.agent import BugFixAgent
from bugfix.code_index import CodeIndex
from bugfix.llm import chat, get_chat_model
from bugfix.sandbox import cleanup
from bugfix.scoring import score_completion
from bugfix.tasks import gold_edit, load_tasks, materialize, read_jsonl, write_jsonl

good = lambda r: r["passed"] and not r["regression"]
asst = lambda text: [{"role": "assistant", "content": text}]


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["task_id"]].append(r)
    return g


def cmd_prompts(a):
    """First-turn prompts for every task, built by the SAME code the agent uses."""
    agent, rows = BugFixAgent(llm=None, index=CodeIndex()), []
    for t in load_tasks(a.tasks):
        ws = materialize(t)
        try:
            msgs, files = agent.build_messages(ws, t["bug_report"], Path(t["repo"]))
        finally:
            cleanup(ws)
        rows.append({"task_id": t["id"], "split": t["split"], "prompt": msgs, "context_files": files})
    write_jsonl(a.out, rows)


def cmd_gold_sft(a):
    tasks = {t["id"]: t for t in load_tasks(a.tasks, "train")}
    rows = []
    for p in read_jsonl(a.prompts):
        t = tasks.get(p["task_id"])
        if not t or t["file"] not in p["context_files"]:  # don't teach editing unseen files
            continue
        ws = materialize(t)
        try:
            gold = gold_edit(t, ws)
        finally:
            cleanup(ws)
        if gold:
            rows.append({"prompt": p["prompt"], "completion": asst(gold)})
    write_jsonl(a.out, rows)


def cmd_sample(a):
    tasks = {t["id"]: t for t in load_tasks(a.tasks, a.split)}
    prompts = [p for p in read_jsonl(a.prompts) if p["task_id"] in tasks]
    random.Random(0).shuffle(prompts)
    prompts = prompts[: a.limit] if a.limit else prompts
    llm = get_chat_model(a.model_name, a.base_url, a.api_key, temperature=a.temperature)

    def work(p):
        try:
            r = chat(llm, p["prompt"])
            return {"task_id": p["task_id"], "prompt": p["prompt"], "completion": r.text,
                    **score_completion(tasks[p["task_id"]], r.text)}
        except Exception as e:
            print("skip:", repr(e))

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(work, [p for p in prompts for _ in range(a.n)]) if r]
    write_jsonl(a.out, rows)
    if rows:
        print(f"pass rate: {sum(good(r) for r in rows) / len(rows):.3f}")


def cmd_sft_from_samples(a):
    """Rejection sampling / response-level distillation: keep verified-correct samples."""
    out = []
    for rs in group(read_jsonl(a.samples)).values():
        seen = set()
        for r in sorted(filter(good, rs), key=lambda r: r["diff_lines"]):
            if r["completion"] in seen:
                continue
            seen.add(r["completion"])
            out.append({"prompt": r["prompt"], "completion": asst(r["completion"])})
            if len(seen) >= a.per_task:
                break
    write_jsonl(a.out, out)


def cmd_dpo(a):
    """Preference pairs from execution: passing+small diff beats failing (hard negatives first)."""
    out = []
    for rs in group(read_jsonl(a.samples)).values():
        chosen, seen = [], set()
        for r in sorted(filter(good, rs), key=lambda r: r["diff_lines"]):
            if r["completion"] not in seen:
                seen.add(r["completion"])
                chosen.append(r)
        rejected = sorted((r for r in rs if not r["passed"]),
                          key=lambda r: (not r["applied"], not r["valid"]))
        for c, j in zip(chosen[: a.pairs_per_task], rejected[: a.pairs_per_task]):
            out.append({"prompt": c["prompt"], "chosen": asst(c["completion"]),
                        "rejected": asst(j["completion"])})
    write_jsonl(a.out, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    def add(name, fn, **defaults):
        p = sub.add_parser(name)
        p.add_argument("--tasks", default="data/tasks.jsonl")
        p.add_argument("--prompts", default="data/prompts.jsonl")
        p.add_argument("--out", required=True)
        p.set_defaults(func=fn)
        return p

    add("prompts", cmd_prompts)
    add("gold-sft", cmd_gold_sft)
    p = add("sample", cmd_sample)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model-name")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    p = add("sft-from-samples", cmd_sft_from_samples)
    p.add_argument("--samples", required=True)
    p.add_argument("--per-task", type=int, default=2)
    p = add("dpo", cmd_dpo)
    p.add_argument("--samples", required=True)
    p.add_argument("--pairs-per-task", type=int, default=2)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()