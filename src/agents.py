from __future__ import annotations
from typing import List, Dict, Optional, TypedDict
import re

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
load_dotenv(os.path.join(parent_dir, ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")


class AgentState(TypedDict, total=False):
    job_description: str
    questions: List[Dict]
    question_idx: int
    latest_answer: Optional[str]
    pending_followups: List[str]
    last_prompt: Optional[str]
    answers: List[Dict]
    done: bool


def find_missing_keywords(answer: str, required_keywords: List[str]) -> List[str]:
    """
    Simple keyword gap checker (deterministic & transparent)
    """
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


def call_llm(openai_api_key, system_prompt, user_prompt):
    client = OpenAI(api_key=openai_api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.2,
        n=1,
        stop=None
    )
    return completion.choices[0].message.content.strip()


class Interviewer:
    def __init__(self):

        builder = StateGraph(AgentState)
        builder.add_node("ask", self.node_ask_question)
        builder.add_node("evaluate", self.node_evaluate_answer)

        # Entry point is the evaluation node – it will immediately route to “ask”
        builder.set_entry_point("evaluate")

        # After evaluation decide where to go next
        builder.add_conditional_edges(
            "evaluate",
            self.router,
            {
                "ask": "ask",
                "end": END,
            })

        # After asking a question we always go back to evaluation (once the UI supplies an answer)
        builder.add_edge("ask", "evaluate")

        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)

    async def run(self, config) -> AgentState:
        start: AgentState = {"job_description": config.job_description, "questions": config.questions,}
        return await self.graph.ainvoke(start, {"configurable": {"thread_id": "ui", "recursion_limit": 100}})

    # Nodes
    async def node_ask_question(self, state: AgentState) -> AgentState:
        """Async ask node that prompts the user and waits for their answer."""
        import asyncio

        if state.get("done"):
            return state

        # Prioritize follow‑ups
        if state.get("pending_followups"):
            prompt = state["pending_followups"].pop(0)
        else:
            question_idx = state.get("question_idx", 0)
            questions = state["questions"]
            if question_idx >= len(questions):
                state["done"] = True
                state["last_prompt"] = None
                return state
            prompt = questions[question_idx]["text"]

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

        question_idx = state.get("question_idx", 0)
        questions = state["questions"]
        if question_idx >= len(questions):
            state["done"] = True
            return state

        latest = (state.get("latest_answer") or "").strip()
        q = questions[question_idx]
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
            state["question_idx"] = question_idx + 1

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


    def router(self, state: AgentState) -> str:
        """Decide the next node based on the current state."""
        if state.get("done"):
            return "end"
        if state.get("pending_followups"):
            return "ask"
        # If we have just recorded an answer (latest_answer cleared) we need to ask next
        return "ask"

