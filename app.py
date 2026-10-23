"""Streamlit chat UI. Run: .venv\\Scripts\\streamlit run app.py"""
import json
import uuid

import streamlit as st

from airport_agent.agent.graph import ask_with_trace, build_agent
from airport_agent.agent.queries import coverage
from airport_agent.voice import transcribe

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")


@st.cache_resource
def get_agent():
    return build_agent()  # one agent (and one in-memory conversation store) per server process


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


if "thread_id" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.header("Airport Investment Agent")
    st.caption("Screens US airports for expansion potential. Scores are computed by deterministic code; "
               "the LLM only picks tools and explains the results.")
    st.button("New conversation", on_click=new_conversation, use_container_width=True)
    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    with st.expander("Data, definitions and limits"):
        info = coverage()
        st.markdown(
            f"- Years: {info['years_available']} (baseline {info['baseline_year']})\n"
            f"- Long haul: nonstop segment of {info['long_haul_miles']:,} miles or more\n"
            f"- Ranked only if at least {info['min_departures_to_rank']:,} departures a year\n"
            f"- Load-factor gate: {info['min_load_factor_gate']:.0%}\n"
            f"- Delay data: months {info['delay_months_sampled']} only\n"
            f"- Expansion score weights: {info['expansion_weights']}"
        )
        st.markdown("\n".join(f"- {text}" for text in info["limitations"]))

st.title("✈️ Airport Investment Intelligence")

try:
    agent = get_agent()
except RuntimeError as error:  # e.g. missing OPENAI_API_KEY
    st.error(str(error))
    st.stop()

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("trace"):
            with st.expander(f"How this was computed ({len(turn['trace'])} tool call(s))"):
                for call in turn["trace"]:
                    st.markdown(f"**{call['name']}** `{json.dumps(call['args'])}`")
                    st.code(call["result"], language="json")

submission = st.chat_input("Ask about US airports, type or use the microphone...", accept_audio=True)
question = st.session_state.pop("pending", None)
if submission:  # a ChatInputValue: .text for typed text, .audio for a recording; None on reruns with no submission
    parts = [submission.text] if submission.text else []
    if submission.audio:
        try:
            with st.spinner("Transcribing..."):
                parts.insert(0, transcribe(submission.audio.getvalue()))
        except RuntimeError as error:
            st.error(str(error))
    question = " ".join(parts) or question
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            try:
                answer, trace = ask_with_trace(agent, question, st.session_state.thread_id)
            except Exception as error:  # show the failure instead of a blank page
                answer, trace = f"Something went wrong: {error}", []
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer, "trace": trace})
    st.rerun()
