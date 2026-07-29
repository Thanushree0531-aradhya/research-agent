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
    """Turn '[1] Title — https://...' lines into clickable markdown links."""
    def repl(match):
        num, title, url = match.group(1), match.group(2), match.group(3)
        return f"[{num}] [{title}]({url})"

    return re.sub(r"\[(\d+)\]\s+(.*?)\s+—\s+(\S+)", repl, text)


if run_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        trace_container = st.container()
        final_container = st.container()

        stage_placeholders = {}
        final_answer = ""

        with trace_container:
            st.subheader("Agent trace")
            for stage_key, label in STAGE_LABELS.items():
                stage_placeholders[stage_key] = st.empty()
                stage_placeholders[stage_key].markdown(f"⬜ {label} — waiting...")

        try:
            for node_name, partial_state in stream_research_agent(question):
                label = STAGE_LABELS.get(node_name, node_name)

                if node_name == "plan":
                    queries = partial_state.get("sub_queries", [])
                    detail = "\n".join(f"- {q}" for q in queries)
                    stage_placeholders[node_name].markdown(f"✅ **{label}**\n{detail}")

                elif node_name == "retrieve":
                    n_sources = len(partial_state.get("sources", []))
                    stage_placeholders[node_name].markdown(
                        f"✅ **{label}** — {n_sources} unique source(s) gathered so far"
                    )

                elif node_name == "analyze":
                    stage_placeholders[node_name].markdown(f"✅ **{label}** — draft written")

                elif node_name == "verify":
                    verified = partial_state.get("verified")
                    notes = partial_state.get("verification_notes", "")
                    icon = "✅" if verified else "⚠️"
                    stage_placeholders[node_name].markdown(
                        f"{icon} **{label}** — verified: `{verified}`"
                        + (f"\n\n*{notes}*" if notes else "")
                    )

                elif node_name == "bump_iteration":
                    stage_placeholders[node_name].markdown(
                        f"🔁 **{label}** — evidence gap found, searching again..."
                    )

                elif node_name == "finalize":
                    final_answer = partial_state.get("final_answer", "")
                    stage_placeholders[node_name].markdown(f"✅ **{label}**")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        if final_answer:
            with final_container:
                st.subheader("Final answer")
                st.markdown(linkify_references(final_answer))