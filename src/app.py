import streamlit as st
from agents import load_job_config, build_graph, initial_state_from_config

st.set_page_config(page_title="AI Job Interviewer", page_icon="🧠", layout="centered")

st.title("🧠 AI Interview Agent (LangGraph + Streamlit)")
st.caption("Asks predefined questions, checks for gaps, and follows up until each answer is complete.")

# --- Sidebar: config
with st.sidebar:
    st.header("Config")
    cfg_path = st.text_input("Job config JSON", value="data/job_config.json")
    if st.button("Reload config"):
        st.session_state.clear()
        st.rerun()

# --- Lazy-init graph + state
if "graph" not in st.session_state:
    config = load_job_config(cfg_path)
    st.session_state.config = config
    st.session_state.graph = build_graph(config)
    st.session_state.state = initial_state_from_config(config)
    # Kick off first question
    st.session_state.state = st.session_state.graph.invoke(st.session_state.state, {"configurable": {"thread_id": "ui"}})  # run 'ask'

# --- Show JD (collapsed)
with st.expander("Job Description", expanded=False):
    st.write(st.session_state.config.job_description)

# --- Chat area
chat = st.container()
with chat:
    # Show history (questions & answers)
    for a in st.session_state.state.get("answers", []):
        st.chat_message("assistant").markdown(f"**Q:** {a['question']}")
        st.chat_message("user").markdown(a["answer"] or "_(no answer)_")
        if a["missing"]:
            st.chat_message("assistant").markdown(
                f"Missing points detected: {', '.join(a['missing'])}"
            )

    # If there is a pending prompt (main or follow-up), show it
    if not st.session_state.state.get("done"):
        current_prompt = st.session_state.state.get("last_prompt")
        if current_prompt:
            st.chat_message("assistant").markdown(current_prompt)
        else:
            # If no last_prompt (first render), ensure we ask one
            st.session_state.state = st.session_state.graph.invoke(st.session_state.state, {"configurable": {"thread_id": "ui"}})
            current_prompt = st.session_state.state.get("last_prompt")
            if current_prompt:
                st.chat_message("assistant").markdown(current_prompt)
    else:
        st.success("✅ Interview complete! Scroll to see the summary below.")

# --- Input
if not st.session_state.state.get("done"):
    user_input = st.chat_input("Type your answer…")
    if user_input is not None:
        # Record user answer into state
        st.session_state.state["latest_answer"] = user_input
        # Evaluate the answer (runs gap-check + generates follow-ups if needed)
        st.session_state.state = st.session_state.graph.invoke(
            st.session_state.state,
            {"configurable": {"thread_id": "ui"}},
            # target="evaluate" not needed if using .invoke with node name; but we compiled with entry ask.
        )
        # After evaluation, either we have follow-ups or move to next question.
        if not st.session_state.state.get("done"):
            # Ask next prompt (follow-up or next question)
            st.session_state.state = st.session_state.graph.invoke(
                st.session_state.state,
                {"configurable": {"thread_id": "ui"}}
            )
        st.rerun()

# --- Summary at the end
if st.session_state.state.get("done"):
    st.subheader("Interview Summary")
    rows = []
    for a in st.session_state.state["answers"]:
        rows.append({
            "Question": a["question"],
            "Missing": ", ".join(a["missing"]) if a["missing"] else "—",
            "Notes": a.get("notes", "")
        })
    if rows:
        st.dataframe(rows, use_container_width=True)
    st.info("Tip: Export your `answers` from `st.session_state.state['answers']` to save a report.")
