# app.py
import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional
import gradio as gr

# --- Import your langgraph builder / Interviewer from local module ---
# You will place the Interviewer and build_graph logic into interview.py (next file).
from .agents import build_graph, initial_state_from_config, load_job_config, config

# Load job config (path relative to working dir)
# cfg_path = "job_config.json"
# config_raw = load_job_config(cfg_path)
# config = JobConfig(
#     jd=config_raw["job_description"],
#     questions=config_raw["questions"],
#     q_idx=0,
#     latest_answer=None,
#     pending_followups=[],
#     last_prompt=None,
#     answers=[],
#     done=False
# )

# Sessions store per-client objects
SESSIONS: Dict[str, Dict[str, Any]] = {}
# Session structure:
# {
#   "graph": <compiled graph>,
#   "state": <AgentState dict>,
#   "pending_prompts": [str, ...],
#   "current_future": asyncio.Future or None,
#   "task": asyncio.Task (running graph)
# }

def create_session() -> str:
    session_id = str(uuid.uuid4())
    graph = build_graph(config)
    # Patch the 'report' node to avoid NameError due to missing imports in agents.py
    async def _dummy_report(state):
        return state
    try:
        graph.update_node("report", _dummy_report)
    except Exception:
        pass
    state = initial_state_from_config(config)
    SESSIONS[session_id] = {
        "graph": graph,
        "state": state,
        "pending_prompts": [],
        "current_future": None,
        "task": None,
        "done": False,
        "last_error": None,
    }
    return session_id

async def get_user_input_factory(session_id: str):
    """
    Returns a coroutine function that the ask node will call:
      prompt_text -> await get_user_input(prompt_text) -> returns user answer string
    It pushes the prompt into pending_prompts so the Gradio UI can pick it up, then awaits the future.
    """
    async def get_user_input(prompt: str) -> str:
        sess = SESSIONS[session_id]
        # push prompt where UI can read it
        sess["pending_prompts"].append(prompt)
        # create a new future and wait for the send_answer to complete it
        fut = asyncio.get_event_loop().create_future()
        sess["current_future"] = fut
        try:
            answer = await fut  # wait until send_answer sets result
            return answer
        finally:
            sess["current_future"] = None
    return get_user_input

async def run_graph_in_background(session_id: str):
    """
    Background task that runs the graph end-to-end. The ask node uses get_user_input to wait for UI replies.
    This task will complete when the graph reaches END.
    """
    sess = SESSIONS[session_id]
    graph = sess["graph"]
    state = sess["state"]

    # produce a fresh get_user_input bound to this session
    get_user_input = await get_user_input_factory(session_id)

    # Replace/patch the "ask" node to call the Interviewer.node_ask_question with our callback.
    # We assume the compiled graph exposes a method get_node_callable(name) or similar.
    # To be robust, we will call the node callable directly if available.
    original_ask_call = None
    try:
        original_ask_call = graph.get_node_callable("ask")
    except Exception:
        # Fallback: graph.update_node will accept a new function
        original_ask_call = None

    async def ask_node_wrapper(s):
        """
        This wrapper will call the original interviewer method but pass get_user_input.
        It preserves your ask logic (follow-ups, q_idx progression) because it calls the same method.
        """
        if original_ask_call is not None:
            # Many LangGraph implementations bind the method; we attempt to call it with the callback
            try:
                return await original_ask_call(s, get_user_input=get_user_input)
            except TypeError:
                # original doesn't accept callback -> call without it (shouldn't happen if you updated Interviewer)
                return await original_ask_call(s)
        else:
            # As a fallback, try to find the Interviewer on a node
            # graph.nodes may be implementation-specific; try to find first node callable
            for n in getattr(graph, "nodes", []):
                node = getattr(graph, "nodes")[n]
                func = getattr(node, "callable", None)
                if func is None:
                    continue
                # if it's the ask function
                if func.__name__ == "node_ask_question":
                    try:
                        return await func(s, get_user_input=get_user_input)
                    except TypeError:
                        return await func(s)
            # hopeless fallback: set last_prompt and return
            s["last_prompt"] = "Could not produce prompt (ask wrapper fallback)"
            return s

    # update the graph node to our wrapper
    try:
        graph.update_node("ask", ask_node_wrapper)
    except Exception:
        # Some LangGraph builds may not provide update_node; in that case we rely on previous logic
        pass

    try:
        # Run the graph until completion
        final_state = await graph.ainvoke(state, {"configurable": {"thread_id": session_id, "recursion_limit": 100}})
        sess["done"] = True
        sess["state"] = final_state
    except Exception as e:
        sess["last_error"] = str(e)
        sess["done"] = True
        # If there's a pending future, set exception so UI unblocks
        fut = sess.get("current_future")
        if fut and not fut.done():
            fut.set_exception(e)
    finally:
        # ensure any leftover future is resolved
        fut = sess.get("current_future")
        if fut and not fut.done():
            fut.set_result("")  # empty answer to let graph proceed or finish

# ---------- Gradio handlers ----------

async def start_session():
    """
    Create a session, start the background graph task, wait for the first prompt (polling briefly),
    and return session_id + first prompt for the UI to show.
    """
    session_id = create_session()
    sess = SESSIONS[session_id]
    # start graph background task
    loop = asyncio.get_event_loop()
    task = loop.create_task(run_graph_in_background(session_id))
    sess["task"] = task

    # Wait for the first prompt to appear in pending_prompts (short poll)
    for _ in range(60):  # wait up to ~6 seconds
        if sess["pending_prompts"]:
            prompt = sess["pending_prompts"].pop(0)
            return session_id, prompt, False  # not done
        if sess.get("last_error"):
            return session_id, f"Error starting session: {sess['last_error']}", True
        await asyncio.sleep(0.1)
    # timed out: still maybe waiting for LLM - return a message
    return session_id, "No prompt yet — the backend is still starting. Try sending a message or wait a moment.", False

async def send_answer(session_id: str, user_text: str):
    """
    Called when the user replies in the chat.
    It completes the session's current_future so the ask node receives the answer.
    Then it waits (polling) for the next prompt or completion, and returns it.
    """
    if session_id not in SESSIONS:
        return "Session not found. Start a new interview.", True

    sess = SESSIONS[session_id]
    if sess.get("last_error"):
        return f"Session error: {sess['last_error']}", True

    # if there is a future waiting, set its result to the user's answer
    fut: Optional[asyncio.Future] = sess.get("current_future")
    if fut is None:
        # There is no ask waiting — perhaps user answered too fast; queue the answer as latest_answer
        # We'll set it into the state directly and let the graph progress next time
        sess["state"]["latest_answer"] = user_text
    else:
        if not fut.done():
            fut.set_result(user_text)

    # wait for next prompt or done
    for _ in range(200):  # wait up to 20 seconds
        if sess["pending_prompts"]:
            prompt = sess["pending_prompts"].pop(0)
            return prompt, False
        if sess.get("done"):
            # graph finished — optionally return final state summary
            return "Interview complete.", True
        if sess.get("last_error"):
            return f"Error: {sess['last_error']}", True
        await asyncio.sleep(0.1)

    return "Still waiting for the next prompt (LLM may be thinking). Try again in a few seconds.", False

# ------------ Gradio UI wiring ------------

with gr.Blocks(title="HR Interview Bot (Gradio)") as demo:
    gr.Markdown("## HR Interview Bot — Gradio\nClick **Start Interview** to begin.")
    start_btn = gr.Button("Start Interview")
    session_id_box = gr.Textbox(label="Session ID (internal)", visible=False)
    chat = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your answer and press Enter")
    done_state = gr.State(value=False)

    async def on_start_click():
        session_id, first_prompt, done = await start_session()
        session_id_box.value = session_id
        # Initialize chat history with the first bot prompt
        chat_history = [("bot", first_prompt)]
        done_state.value = done
        return session_id_box, chat_history

    start_btn.click(on_start_click, outputs=[session_id_box, chat])

    async def on_user_submit(text, session_id, chat_history, done_flag):
        # if session missing, inform user
        if not session_id:
            return chat_history + [("bot", "Start an interview first.")], session_id, done_flag
        prompt, done = await send_answer(session_id, text)
        # append user text and bot prompt
        chat_history = chat_history + [("user", text)]
        chat_history = chat_history + [("bot", prompt)]
        done_flag = done
        return chat_history, session_id, done_flag

    user_input.submit(on_user_submit, inputs=[user_input, session_id_box, chat, done_state], outputs=[chat, session_id_box, done_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
