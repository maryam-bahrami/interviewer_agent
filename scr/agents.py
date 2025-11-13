from __future__ import annotations
from typing import List, Dict, Optional, TypedDict
from dataclasses import dataclass
import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ---- Types --------------------------------------------------------------

class AgentState(TypedDict, total=False):
    jd: str
    questions: List[Dict]
    q_idx: int
    latest_answer: Optional[str]
    pending_followups: List[str]
    last_prompt: Optional[str]
    answers: List[Dict]
    done: bool

# ---- Config loader -------------------------------------------------------

@dataclass
class JobConfig:
    job_description: str
    questions: List[Dict]

def load_job_config(path: str | Path) -> JobConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return JobConfig(job_description=data["job_description"], questions=data["questions"])

# ---- Helper --------------------------------------------------------------


# ---- Interviewer class ---------------------------------------------------

class Interviewer:
    """Encapsulates the ask/evaluate logic as class methods."""

    async def node_ask_question(self, state: AgentState) -> AgentState:
        """Async ask node that prompts the user and waits for their answer."""
        import asyncio

        if state.get("done"):
            return state

        # Prioritize follow‑ups
        if state.get("pending_followups"):
            prompt = state["pending_followups"].pop(0)
        else:
            q_idx = state.get("q_idx", 0)
            questions = state["questions"]
            if q_idx >= len(questions):
                state["done"] = True
                state["last_prompt"] = None
                return state
            prompt = questions[q_idx]["text"]

        # Show the prompt and wait for user input asynchronously
        print(prompt)
        answer = await asyncio.to_thread(lambda: input("> "))
        # Store the answer so the evaluation node can process it
        state["latest_answer"] = answer
        state["last_prompt"] = prompt
        print("*****")
        print(state)
        print("*****")
        return state

    def node_evaluate_answer(self, state: AgentState) -> AgentState:
        """Evaluate the latest answer, generate follow‑ups, and advance the state."""
        if state.get("done"):
            return state

        # If no answer has been provided yet (initial entry), just skip processing.
        if state.get("latest_answer") is None:
            return state

        q_idx = state.get("q_idx", 0)
        questions = state["questions"]
        if q_idx >= len(questions):
            state["done"] = True
            return state

        latest = (state.get("latest_answer") or "").strip()
        q = questions[q_idx]
        required = q.get("required_keywords", [])
        missing = self.find_missing_keywords(latest, required)

        # Record the answer
        answers = state.get("answers", [])
        answers.append({
            "question_id": q["id"],
            "question": q["text"],
            "answer": latest,
            "missing": missing,
            "notes": q.get("guidance", "")
        })
        state["answers"] = answers

        # Generate follow‑ups for missing keywords
        if missing:
            for m in missing:
                fu = f"You didn’t mention “{m}”. Could you add details regarding {m}?"
                state.setdefault("pending_followups", []).append(fu)
        else:
            # Advance to the next main question
            state["q_idx"] = q_idx + 1

        # Reset the latest answer for the next round
        state["latest_answer"] = None
        return state

    def find_missing_keywords(self, answer: str, required_keywords: List[str]) -> List[str]:
        """Return a list of required keywords that are not present in *answer*."""
        if not required_keywords:
            return []
        a = answer.lower()
        missing = []
        for kw in required_keywords:
            pattern = r"\b" + re.escape(kw.lower()).replace(r"\ ", r"[\s\-/\\_]+") + r"\b"
            if not re.search(pattern, a):
                missing.append(kw)
        return missing

# ---- Router --------------------------------------------------------------

    def router(self, state: AgentState) -> str:
        """Decide the next node based on the current state."""
        if state.get("done"):
            return "end"
        if state.get("pending_followups"):
            return "ask"
        # If we have just recorded an answer (latest_answer cleared) we need to ask next
        return "ask"

# ---- Graph builder -------------------------------------------------------

def build_graph(config: JobConfig) -> StateGraph:
    """Create a LangGraph where the interview flow is driven entirely by the graph."""
    interviewer = Interviewer()

    builder = StateGraph(AgentState)

    # Register the class methods as graph nodes
    builder.add_node("ask", interviewer.node_ask_question)
    builder.add_node("evaluate", interviewer.node_evaluate_answer)

    # Entry point is the evaluation node – it will immediately route to “ask”
    builder.set_entry_point("evaluate")

    # After evaluation decide where to go next
    builder.add_conditional_edges(
        "evaluate",
        interviewer.router,
        {
            "ask": "ask",
            "end": END,
        },
    )

    # After asking a question we always go back to evaluation (once the UI supplies an answer)
    builder.add_edge("ask", "evaluate")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

# ---- Helpers to seed initial state -----------------------------------------

def initial_state_from_config(config: JobConfig) -> AgentState:
    """Create the initial LangGraph state from the loaded job configuration."""
    return AgentState(
        jd=config.job_description,
        questions=config.questions,
        q_idx=0,
        latest_answer=None,
        pending_followups=[],
        last_prompt=None,
        answers=[],
        done=False,
    )
