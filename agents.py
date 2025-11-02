from __future__ import annotations
from typing import List, Dict, Optional, TypedDict
from dataclasses import dataclass
import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ---- Types

class AgentState(TypedDict, total=False):
    jd: str
    questions: List[Dict]
    q_idx: int
    latest_answer: Optional[str]
    pending_followups: List[str]
    last_prompt: Optional[str]
    answers: List[Dict]  # {question_id, question, answer, missing, notes}
    done: bool

# ---- Config loader

@dataclass
class JobConfig:
    job_description: str
    questions: List[Dict]

def load_job_config(path: str | Path) -> JobConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return JobConfig(job_description=data["job_description"], questions=data["questions"])

# ---- Simple keyword gap checker (deterministic & transparent)

def find_missing_keywords(answer: str, required_keywords: List[str]) -> List[str]:
    if not required_keywords:
        return []
    a = answer.lower()
    missing = []
    for kw in required_keywords:
        # Accept keyword match if any token group appears (rough, but robust)
        # Also tolerate separators like / , - _
        pattern = r"\b" + re.escape(kw.lower()).replace(r"\ ", r"[\s\-/\\_]+") + r"\b"
        if not re.search(pattern, a):
            missing.append(kw)
    return missing

# ---- Node functions

def node_ask_question(state: AgentState) -> AgentState:
    """Select next prompt to ask: follow-up if pending, else main question."""
    if state.get("done"):
        return state

    # Prioritize follow-ups
    if state.get("pending_followups"):
        prompt = state["pending_followups"].pop(0)
        state["last_prompt"] = prompt
        return state

    q_idx = state.get("q_idx", 0)
    questions = state["questions"]
    if q_idx >= len(questions):
        state["done"] = True
        state["last_prompt"] = None
        return state

    state["last_prompt"] = questions[q_idx]["text"]
    return state

def node_evaluate_answer(state: AgentState) -> AgentState:
    """Evaluate user's latest answer and decide if we need follow-ups or can advance."""
    if state.get("done"):
        return state
    q_idx = state.get("q_idx", 0)
    questions = state["questions"]
    if q_idx >= len(questions):
        state["done"] = True
        return state

    latest = (state.get("latest_answer") or "").strip()
    q = questions[q_idx]
    required = q.get("required_keywords", [])
    missing = find_missing_keywords(latest, required)

    # Save answer snapshot
    answers = state.get("answers", [])
    answers.append({
        "question_id": q["id"],
        "question": q["text"],
        "answer": latest,
        "missing": missing,
        "notes": q.get("guidance", "")
    })
    state["answers"] = answers

    # Generate simple follow-ups for each missing keyword
    if missing:
        for m in missing:
            fu = f"You didn’t mention “{m}”. Could you add details regarding {m}?"
            state.setdefault("pending_followups", []).append(fu)
    else:
        # Move to next main question
        state["q_idx"] = q_idx + 1

    # Reset the latest answer (will be set again by UI)
    state["latest_answer"] = None
    return state

def router(state: AgentState) -> str:
    if state.get("done"):
        return "end"
    # If there are follow-ups to ask after an evaluation, ask them next
    if state.get("pending_followups"):
        return "ask"
    # If we just advanced to next question, ask it
    q_idx = state.get("q_idx", 0)
    if q_idx < len(state["questions"]):
        # If we've just recorded an answer, go ask the next question
        return "ask"
    # No more questions → end
    return "end"

# ---- Graph builder

def build_graph(config: JobConfig):
    builder = StateGraph(AgentState)

    builder.add_node("ask", node_ask_question)
    builder.add_node("evaluate", node_evaluate_answer)

    # Entry: ask first question
    builder.set_entry_point("ask")

    # After user answers, we call evaluate, then route
    builder.add_conditional_edges("evaluate", router, {
        "ask": "ask",
        "end": END
    })

    # From ask, the UI asks the user. After receiving an answer, UI will call "evaluate".
    # So we don't auto-connect ask->evaluate; the UI controls when to run "evaluate".

    memory = MemorySaver()  # optional checkpointing
    graph = builder.compile(checkpointer=memory)
    return graph

# ---- Helpers to seed initial state

def initial_state_from_config(config: JobConfig) -> AgentState:
    return AgentState(
        jd=config.job_description,
        questions=config.questions,
        q_idx=0,
        latest_answer=None,
        pending_followups=[],
        last_prompt=None,
        answers=[],
        done=False
    )
