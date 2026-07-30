"""
Streamlit UI for the Research Scientist Agent.

Run with:
    streamlit run app.py
"""

import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import stream_research_agent  # noqa: E402

st.set_page_config(page_title="Research Scientist Agent", page_icon="🔬", layout="centered")

st.title("🔬 Research Scientist Agent")

question = st.text_input(
    "Research question",
    placeholder="e.g. Compare RAG and Fine-tuning",
)
run_clicked = st.button("Run research", type="primary", use_container_width=True)

STAGE_LABELS = {
    "plan": "🧭 Plan",
    "retrieve": "🔎 Retrieve",
    "analyze": "🧠 Analyze",
    "verify": "✅ Verify",
    "bump_iteration": "🔁 Looping back for more evidence",
    "finalize": "📄 Finalize",
}


def linkify_references(text: str) -> str:
    def repl(match):
        num, title, url = match.group(1), match.group(2), match.group(3)
        return f"[{num}] [{title.strip()}]({url})"

    return re.sub(r"\[(\d+)\]\s+(.*)\s+—\s+(https?://\S+)", repl, text)


def score_color(score):
    if score >= 0.8:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


if run_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        trace_container = st.container()
        final_container = st.container()

        stage_placeholders = {}
        stages_reached = set()

        final_answer = ""
        draft_answer = ""
        support_ratio = None
        eval_scores = None
        retrieved_sources = []
        unsupported_claims = []

        with trace_container:
            st.subheader("Agent trace")
            for stage_key, label in STAGE_LABELS.items():
                stage_placeholders[stage_key] = st.empty()
                stage_placeholders[stage_key].markdown(f"⬜ {label} — waiting...")

        try:
            for node_name, partial_state in stream_research_agent(question):
                label = STAGE_LABELS.get(node_name, node_name)
                stages_reached.add(node_name)

                if node_name == "plan":
                    queries = partial_state.get("sub_queries", [])
                    detail = "\n".join(f"- {q}" for q in queries)
                    stage_placeholders[node_name].markdown(f"✅ **{label}**\n{detail}")

                elif node_name == "retrieve":
                    retrieved_sources = partial_state.get("sources", [])
                    n_sources = len(retrieved_sources)

                    stage_placeholders[node_name].markdown(
                        f"✅ **{label}** — {n_sources} unique source(s) gathered"
                    )

                    # ✅ SHOW CONTEXTS (VERY IMPORTANT)
                    if retrieved_sources:
                        with st.expander("📚 Retrieved Context (Top 5)"):
                            for i, src in enumerate(retrieved_sources[:5]):
                                st.markdown(f"**[{i+1}]** {str(src)[:300]}...")

                elif node_name == "analyze":
                    stage_placeholders[node_name].markdown(f"✅ **{label}** — draft written")

                elif node_name == "verify":
                    verified = partial_state.get("verified")
                    notes = partial_state.get("verification_notes", "")
                    support_ratio = partial_state.get("support_ratio")
                    unsupported_claims = partial_state.get("unsupported_claims", []) or []

                    icon = "✅" if verified else "⚠️"

                    if support_ratio is not None:
                        support_detail = f"(claim support: {support_ratio:.0%})"
                    else:
                        support_detail = "(⚠️ verifier failed — check format)"

                    stage_placeholders[node_name].markdown(
                        f"{icon} **{label}** — verified: `{verified}` {support_detail}"
                        + (f"\n\n*{notes}*" if notes else "")
                    )

                elif node_name == "bump_iteration":
                    iteration = partial_state.get("iteration")
                    stage_placeholders[node_name].markdown(
                        f"🔁 **{label}** (iteration {iteration}) — refining search..."
                    )

                elif node_name == "finalize":
                    final_answer = partial_state.get("final_answer", "")
                    draft_answer = partial_state.get("draft_answer", "")
                    eval_scores = partial_state.get("eval_scores")
                    stage_placeholders[node_name].markdown(f"✅ **{label}**")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        # mark skipped stages
        for stage_key, label in STAGE_LABELS.items():
            if stage_key not in stages_reached:
                stage_placeholders[stage_key].markdown(f"⏭️ {label} — not needed")

        if final_answer:
            with final_container:
                st.subheader("Final answer")
                st.markdown(linkify_references(final_answer))

                # ✅ RAW ANSWER DEBUG — the draft *before* reference list,
                # RAGAS scores, and any unsupported-claim stripping notes
                # were appended. Useful to see what the LLM actually wrote
                # vs. what post-processing added on top of it.
                with st.expander("🧾 Raw Answer (pre-reference draft)"):
                    st.text(draft_answer if draft_answer else "(no draft captured)")

                # ✅ UNSUPPORTED CLAIMS DEBUG — what verification flagged
                # and finalize_node stripped out before scoring.
                if unsupported_claims:
                    with st.expander(f"🚫 Unsupported Claims Removed ({len(unsupported_claims)})"):
                        for c in unsupported_claims:
                            citation = c.get("citation")
                            claim_text = c.get("claim", "")
                            cite_str = f" [{citation}]" if citation else " [no citation]"
                            st.markdown(f"- {claim_text}{cite_str}")

                # ✅ CONTEXT COUNT
                st.metric("Contexts Used", len(retrieved_sources))

                # --- QUALITY METRICS ---
                st.subheader("📊 Answer Quality")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if eval_scores and "faithfulness" in eval_scores:
                        val = eval_scores["faithfulness"]
                        st.metric("Faithfulness", f"{score_color(val)} {val:.2f}")
                    else:
                        st.metric("Faithfulness", "N/A")

                with col2:
                    if eval_scores and "answer_relevancy" in eval_scores:
                        val = eval_scores["answer_relevancy"]
                        st.metric("Answer Relevancy", f"{score_color(val)} {val:.2f}")

                        if val < 0.7:
                            st.warning("⚠️ Low relevancy — improve retrieval or prompt")
                    else:
                        st.metric("Answer Relevancy", "N/A")

                with col3:
                    if support_ratio is not None:
                        st.metric("Claim Support", f"{support_ratio:.0%}")
                    else:
                        st.metric("Claim Support", "Unknown")

                # ✅ DEBUG EVAL SCORES
                if eval_scores:
                    with st.expander("📊 Full Eval Scores (Debug)"):
                        st.json(eval_scores)
                else:
                    st.warning("⚠️ eval_scores is None — RAGAS didn't run")

                # warnings
                if support_ratio is None:
                    st.caption("⚠️ Verifier failed → claims not validated properly")
                elif support_ratio < 0.8:
                    st.caption("⚠️ Some claims were weakly supported and removed")

        else:
            with final_container:
                st.error(
                    "The agent finished but did not produce a final answer. "
                    "Check trace above."
                )