import argparse
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bugfix.agent import BugFixAgent
from bugfix.code_index import CodeIndex
from bugfix.llm import get_chat_model
from bugfix.sandbox import cleanup
from bugfix.tasks import load_tasks, materialize, write_jsonl

RESULTS = Path("results")


def run_one(agent, task):
    ws = materialize(task)
    try:
        row = agent.solve(ws, task["bug_report"], repo_root=Path(task["repo"]),
                          gold_file=task["file"], n_total=task["n_total"])
    except Exception as exc:  # one failure shouldn't kill the run
        row = {"success": False, "error": repr(exc), "attempts_log": []}
    finally:
        cleanup(ws)
    row["task_id"] = task["id"]
    return row


def bootstrap_ci(values, n=1000):
    rng, m = random.Random(0), len(values)
    means = sorted(sum(rng.choices(values, k=m)) / m for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)


def summarize(rows, meta):
    first = [float(bool(r["attempts_log"]) and r["attempts_log"][0]["passed"]) for r in rows]
    final = [float(r["success"]) for r in rows]
    applied = [float(bool(r["attempts_log"]) and r["attempts_log"][0]["applied"]) for r in rows]
    regress = [float(r.get("regression", False)) for r in rows]
    tokens = [r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rows]
    recall = [float(r["gold_in_context"]) for r in rows if r.get("gold_in_context") is not None]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    return {
        **meta, "n_tasks": len(rows),
        "success@1": mean(first), "success@1_ci95": bootstrap_ci(first),
        "success@k": mean(final), "success@k_ci95": bootstrap_ci(final),
        "patch_validity@1": mean(applied), "regression_rate": mean(regress),
        "avg_attempts": mean([len(r["attempts_log"]) for r in rows]),
        "retrieval_recall": mean(recall),
        "avg_tokens": round(statistics.mean(tokens)),
        "median_latency_s": round(statistics.median([r.get("latency_s", 0) for r in rows]), 2),
        "successes_per_1k_tokens": round(sum(final) / max(sum(tokens), 1) * 1000, 4),
    }


def compare(tags):
    cols = ["success@1", "success@k", "patch_validity@1", "regression_rate",
            "avg_attempts", "avg_tokens", "median_latency_s", "retrieval_recall"]
    print("| tag | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for t in tags:
        s = json.loads((RESULTS / f"{t}_summary.json").read_text())
        print(f"| {t} | " + " | ".join(str(s[c]) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-name")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)

    tasks = load_tasks(a.tasks, a.split)
    random.Random(0).shuffle(tasks)
    tasks = tasks[: a.limit]
    llm = get_chat_model(a.model_name, a.base_url, a.api_key, temperature=0.0)
    agent = BugFixAgent(llm, CodeIndex(), max_attempts=a.attempts)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda t: run_one(agent, t), tasks))

    RESULTS.mkdir(exist_ok=True)
    write_jsonl(RESULTS / f"{a.tag}.jsonl", rows)
    summary = summarize(rows, {"tag": a.tag, "model": a.model_name, "split": a.split,
                               "max_attempts": a.attempts})
    (RESULTS / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()