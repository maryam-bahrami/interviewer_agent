import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

parent_dir = Path(__file__).resolve().parent.parent
load_dotenv(parent_dir / ".env")

st.set_page_config(page_title="Interview Agent", page_icon="🎤")
st.title("🎤 Interviewer Assistant")
st.markdown("Bot asks questions → you answer → it evaluates → follow-ups if needed → final report. 🚀")



from agents import (
    config,
    initial_state_from_config,
    Interviewer,
    get_next_prompt,
)

def push(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})


def show_current_chat():
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def reset():
    st.session_state.state = dict(initial_state_from_config(config))
    st.session_state.chat = []
    st.session_state.started = False
    st.session_state.report_ready = False
    st.session_state.last_shown_prompt = None


if "interviewer" not in st.session_state:
    st.session_state.interviewer = Interviewer()

if "state" not in st.session_state:
    st.session_state.state = dict(initial_state_from_config(config))

if "chat" not in st.session_state:
    st.session_state.chat = []

if "started" not in st.session_state:
    st.session_state.started = False

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

if "last_shown_prompt" not in st.session_state:
    st.session_state.last_shown_prompt = None


with st.sidebar:
    st.header("Controls")

    if st.button("🔄 Reset interview", use_container_width=True):
        reset()
        st.rerun()

    st.divider()
    st.subheader("Progress")
    q_idx = st.session_state.state.get("q_idx", 0)
    total = len(st.session_state.state.get("questions", []))
    pending = len(st.session_state.state.get("pending_followups", []))
    done = st.session_state.state.get("done", False)

    st.write(f"Question: {min(q_idx + 1, total)} / {total}")
    st.write(f"Follow-ups queued: {pending}")
    st.write(f"Done: {done}")

    st.divider()
    show_debug = st.toggle("Show debug feedback (LLM JSON)", value=False)

    now = datetime.now()

    st.header("🗓️ Today")
    st.write(f"**Date:** {now:%A, %d %B %Y}")
    st.write(f"**Time:** {now:%H:%M:%S}")

    # support button
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("If you encounter any issue during interview, Please **contact us**:", unsafe_allow_html=True)

    recipient_email = "support@example.com"
    subject = "Support Request"
    body = "Please describe the issue or question you're experiencing:"
    mailto_link = f"mailto:{recipient_email}?subject={subject}&body={body}"
    st.link_button(label="✉ Contact Support", url=mailto_link)
    st.markdown("---")
    st.caption("Built for internal HR policy demos with with Streamlit · Powered by OpenAI ✨")


show_current_chat()

if not st.session_state.started:
    if st.button("🚀 Start interview", use_container_width=True):
        st.session_state.started = True

        prompt = get_next_prompt(st.session_state.state)
        if prompt:
            # If it's a follow-up, pop it when we display it
            if st.session_state.state.get("pending_followups"):
                st.session_state.state["pending_followups"].pop(0)

            st.session_state.last_shown_prompt = prompt
            push("assistant", prompt)

        st.rerun()
    st.stop()

state = st.session_state.state

if state.get("done") and not st.session_state.report_ready:
    with st.spinner("Generating final review and report..."):
        state = st.session_state.interviewer.node_reviewer(state)
        state = st.session_state.interviewer.node_reporter(state)
        st.session_state.state = state

    st.session_state.report_ready = True
    push("assistant", "✅ Interview complete. I generated your report below.")
    st.rerun()


# ------------------------
# Show report if ready
# ------------------------
if st.session_state.report_ready and st.session_state.state.get("report"):
    st.divider()
    st.subheader("📄 Candidate Evaluation Report")
    st.markdown(st.session_state.state["report"])

    st.download_button(
        "⬇️ Download report.md",
        data=st.session_state.state["report"].encode("utf-8"),
        file_name="report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # Optional: show raw review JSON/text
    with st.expander("Show raw reviewer output"):
        st.write(st.session_state.state.get("review", ""))

    st.stop()


# ------------------------
# Optional debug feedback (per question evaluation JSON history)
# ------------------------
if show_debug:
    with st.expander("Debug: LLM evaluation history (per question)", expanded=False):
        llm_responses = st.session_state.state.get("llm_responses", []) or []
        st.write(llm_responses)


# ------------------------
# Normal turn: user answers
# ------------------------
user_text = st.chat_input("Type your answer and press Enter...")

if user_text:
    # 1) show user message
    push("user", user_text)

    # 2) bind prompt + answer into state for evaluation
    st.session_state.state["last_prompt"] = st.session_state.last_shown_prompt
    st.session_state.state["latest_answer"] = user_text

    # 3) evaluate
    with st.spinner("Evaluating..."):
        st.session_state.state = st.session_state.interviewer.node_evaluate_answer(
            st.session_state.state
        )

    # 4) compute next prompt (follow-up or next question)
    next_prompt = get_next_prompt(st.session_state.state)

    if not next_prompt:
        # mark done and let the report generation branch handle it next rerun
        st.session_state.state["done"] = True
        push("assistant", "Thanks! That’s all questions. Preparing your report…")
        st.rerun()

    # If it is a follow-up prompt, pop it *now* since we are about to show it.
    if st.session_state.state.get("pending_followups"):
        # get_next_prompt returns pending_followups[0] when present
        st.session_state.state["pending_followups"].pop(0)

    st.session_state.last_shown_prompt = next_prompt
    push("assistant", next_prompt)

    st.rerun()
