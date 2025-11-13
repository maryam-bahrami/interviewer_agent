from __future__ import annotations
from typing import List, Dict, Optional, TypedDict
from dataclasses import dataclass
import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
load_dotenv(os.path.join(parent_dir, ".env"))



class AgentState(TypedDict, total=False):
    jd: str
    questions: List[Dict]
    q_idx: int
    latest_answer: Optional[str]
    pending_followups: List[str]
    last_prompt: Optional[str]
    answers: List[Dict]
    done: bool


def call_llm(system_prompt, user_prompt):
    openai_api_key = os.getenv("OPENAI_API_KEY")
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

    def reviewer_node(self, question, answer, expected_points):
        system_prompt = """
        You are an expert answer reviewer.
        Your job:
        - Evaluate how well a candidate's answer responds to a given question.
        - Compare the answer against a list of expected points or keywords.
        - Be strict but fair, and explain your reasoning briefly.
        - Never invent facts that are not in the answer.
        
        Evaluation criteria:
        1. Relevance – Does the answer actually address the question?
        2. Completeness – How many of the expected points are covered?
        3. Depth – Does the answer show understanding, not just buzzwords?
        4. Clarity – Is the answer clear and coherent?
        
        Return your evaluation **only** in this JSON format:
        
        {
          "score": 0–100,
          "verdict": "excellent" | "good" | "average" | "poor",
          "covered_points": [ "point1", "point2", ... ],
          "missing_points": [ "pointX", "pointY", ... ],
          "strengths": "short paragraph",
          "weaknesses": "short paragraph",
          "follow_up_question": "one concise follow-up question focusing on gaps"
        }
        
        If something is not applicable, use an empty list or an empty string.
        Do not include any other text outside the JSON.
        """

        user_prompt = f"""
        Question:
        {question}
        
        Candidate answer:
        {answer}
        
        Expected points to look for (can be keywords, concepts, or examples):
        {expected_points}
        
        Please review the answer based on the system instructions and return the JSON evaluation.
        """

        llm_response = call_llm(system_prompt, user_prompt)
        return llm_response

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
