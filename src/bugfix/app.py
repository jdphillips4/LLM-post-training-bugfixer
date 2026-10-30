import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bugfix import config
from bugfix.agent import BugFixAgent
from bugfix.code_index import CodeIndex
from bugfix.llm import get_chat_model
from bugfix.sandbox import cleanup, make_workspace

st.set_page_config(page_title="Bug-fix agent", layout="wide")


@st.cache_resource
def get_index():
    return CodeIndex()


run_tab, results_tab = st.tabs(["Fix a bug", "Experiment results"])

with run_tab:
    repo = st.text_input("Repository path (must have a pytest suite)", "repos/boltons")
    report = st.text_area("Bug report", height=140)
    c1, c2, c3 = st.columns(3)
    model = c1.text_input("Model name", config.MODEL_NAME)
    base_url = c2.text_input("Base URL", config.BASE_URL)
    attempts = c3.slider("Max attempts", 1, 5, 3)

    if st.button("Run agent") and repo and report:
        ws = make_workspace(Path(repo))  # works on a copy; your repo is never modified
        try:
            with st.spinner("Running agent and tests..."):
                agent = BugFixAgent(get_chat_model(model, base_url), get_index(), max_attempts=attempts)
                traj = agent.solve(ws, report, repo_root=Path(repo), append_test_output=True)
        finally:
            cleanup(ws)
        (st.success if traj["success"] else st.error)(
            f"{'Fixed' if traj['success'] else 'Not fixed'} after {traj.get('attempts', 0)} attempt(s) "
            f"| {traj['prompt_tokens'] + traj['completion_tokens']} tokens | {traj['latency_s']:.1f}s")
        st.code(traj["diff"] or "(no applied patch)", language="diff")
        with st.expander("Attempt log"):
            st.json(traj["attempts_log"])
        with st.expander("Full conversation"):
            for m in traj["messages"]:
                st.markdown(f"**{m['role']}**")
                st.text(m["content"])

with results_tab:
    files = sorted(Path("results").glob("*_summary.json"))
    if files:
        df = pd.DataFrame([json.loads(f.read_text()) for f in files]).set_index("tag")
        st.dataframe(df[["model", "n_tasks", "success@1", "success@k", "patch_validity@1",
                         "regression_rate", "avg_attempts", "avg_tokens", "median_latency_s"]])
        st.bar_chart(df[["success@1", "success@k"]])
    else:
        st.info("No results yet. Run `python -m bugfix.evaluate` first.")