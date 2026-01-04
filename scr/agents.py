from __future__ import annotations
from typing import List, Dict, Optional, TypedDict
from dataclasses import dataclass
import json
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
load_dotenv(os.path.join(parent_dir, ".env"))

def load_job_config(cfg_path):
  with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class JobConfig:
    jd: str
    questions: list
    q_idx: int
    latest_answer: list
    pending_followups: list
    last_prompt: str
    answers: list
    no_followup_chances: int
    done: bool

# Path to the job configuration JSON file (relative to this script)
cfg_path = os.getenv("JOB_DESCRIPTION_PATH")

# Load job description and interview questions
config = load_job_config(cfg_path)

config = JobConfig(
    jd=config["job_description"],
    questions=config["questions"],
    q_idx=0,  # Start state
    latest_answer=None,
    pending_followups=[],
    last_prompt=None,
    answers=[],
    no_followup_chances=int(config["number_of_followup_chances"]),
    done=False
)


class AgentState(TypedDict, total=False):
    jd: str
    questions: List[Dict]
    q_idx: int
    latest_answer: Optional[str]
    pending_followups: List[str]
    last_prompt: Optional[str]
    answers: List[Dict]
    llm_responses: List[Dict]
    done: bool
    report: str
    review: str

def initial_state_from_config(config: JobConfig) -> AgentState:
    """Create the initial LangGraph state from the loaded job configuration."""
    return AgentState(
        jd=config.jd,
        questions=config.questions,
        q_idx=0,
        latest_answer=None,
        pending_followups=[],
        last_prompt=None,
        answers=[],
        llm_responses=[],
        review = "",
        done=False,
    )

def initial_state_from_config(config: JobConfig) -> AgentState:
    """Create the initial LangGraph state from the loaded job configuration."""
    return AgentState(
        jd=config.jd,
        questions=config.questions,
        q_idx=0,
        latest_answer=None,
        pending_followups=[],
        last_prompt=None,
        answers=[],
        llm_responses=[],
        review = "",
        done=False,
    )

class Interviewer:
    """Encapsulates the ask/evaluate logic as class methods."""

    def __init__(self):
        self.chat_model = init_chat_model(
                                        os.getenv("MODEL_NAME", "gpt-4o-mini"),
                                        base_url=os.getenv("BASE_URL"),
                                        api_key=os.getenv("API_KEY", "not-needed"),
                                           )

    async def node_ask_question(self, state: AgentState, get_user_input=None) -> AgentState:
        if state.get("done"):
            return state

        # --- PRIORITIZE FOLLOW-UPS (unchanged) ---
        if state["pending_followups"]:
            prompt = state["pending_followups"].pop(0)
        else:
            q_idx = state["q_idx"]
            if q_idx >= len(state["questions"]):
                state["done"] = True
                return state
            prompt = state["questions"][q_idx]["text"]

        # ---- FRONTEND INPUT INSTEAD OF CONSOLE INPUT ----
        if get_user_input is None:
            # fallback: console-based
            import asyncio
            answer = await asyncio.to_thread(lambda: input("> "))
        else:
            # frontend-based
            answer = await get_user_input(prompt)

        state["latest_answer"] = answer
        state["last_prompt"] = prompt
        return state


        return state

    def node_evaluate_answer(self, state: AgentState) -> AgentState:
        """Evaluate the latest answer, generate follow-ups, and advance the state."""
        if state.get("done"):
            return state
    
        if state.get("latest_answer") is None:
            return state
    
        q_idx = state.get("q_idx", 0)
        questions = state["questions"]
        if q_idx >= len(questions):
            state["done"] = True
            return state
    
        latest = (state.get("latest_answer") or "").strip()
        q = questions[q_idx]
        question = q["text"]
        required_keywords = q.get("required_keywords", [])
    
        # ----- LLM CALL -----------------------------------------------------------
        system_prompt = """
        You are an expert hiring assistant evaluating candidate answers during an interview. Your task is to check whether the candidate’s answer demonstrates understanding of the topic. Understanding may be shown in two ways: 1. The answer explicitly contains the expected keywords. 2. The answer does not use the exact keywords but explains the concepts correctly and completely. Always evaluate based on meaning, not just exact wording. Return your result in the specified JSON structure and be kind with the user.
        Return ONLY valid JSON.
        """
    
        user_prompt = """
        [Interview Question]
        {question}
    
        [Expected Keywords or Concepts]
        {required_keywords}
    
        [Candidate Answer]
        {latest}
    
        Your task: 
        1. Identify whether the candidate’s answer contains each expected keyword. 
        2. If a keyword is missing but the candidate clearly explains the idea, mark it as "explained". 
        3. If neither the keyword nor the concept is present, mark it as "missing". 
        4. Give a short explanation for each classification. 
        5. If a keyword is missing make one follow-up question to clarify the missing keywords in "follow_up" 
        6. If the user does not understand the question, try to clarify and reformulate the question.
        7. If the user asks question about company data, do not provide the the data and reject the user's request respectfully.
        8. If the user provides any personal data (e.g age, gender, marital status, address etc.), do not store this data in the memory.
        9. Do not ask user any discriminative questions (e.g gender, nationality, color of skin etc.)
        10. If a question was answered previously, do not ask it again. 
        
        JSON format:
        {{
          "keywords": [
            {{
              "keyword": "",
              "status": "present | explained | missing",
              "explanation": ""
            }}
          ],
          "overall_assessment": "",
          "score": "",
          "follow_up": ""
        }}
        """

        user_prompt = user_prompt.format(question = q["text"],
                                           required_keywords=required_keywords,
                                           latest=latest)
        
        llm_response = self.chat_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        parsed = json.loads(llm_response.content)
    
        # ---- TRACK LLM RESPONSES -------------------------------------------------
        if "llm_responses" not in state:
            state["llm_responses"] = []
    
        # initialize response tracker for this question if needed
        if len(state["llm_responses"]) <= q_idx:
            state["llm_responses"].append({
                "q_idx": q_idx,
                "follow_up_count": 0,
                "history": []
            })
    
        q_track = state["llm_responses"][q_idx]
        q_track["history"].append(parsed)
    
        follow_up = parsed.get("follow_up", "").strip()
    
        # ---- FOLLOW-UP LOGIC WITH LIMIT -------------------------------------
        if follow_up:  
            if q_track["follow_up_count"] < config.no_followup_chances :
                q_track["follow_up_count"] += 1
                state["pending_followups"].append(follow_up)
            else:
                # exceeded 3 follow-ups → move to next question
                #print("⚠️ Maximum follow-ups reached. Moving to next question.")
                state["pending_followups"] = []
                state["q_idx"] = q_idx + 1
        else:
            # answer was complete → move to next question
            state["q_idx"] = q_idx + 1
    
        # ---- SAVE USER ANSWER ----------------------------------------------------
        answers = state.get("answers", [])
        answers.append({
            "question_id": q["id"],
            "question": q["text"],
            "answer": latest,
            "notes": q.get("guidance", "")
        })
        state["answers"] = answers
    
        #state["latest_answer"] = None

        return state
            
    def node_reviewer(self, state: AgentState) -> AgentState:
        reviewer_prompt = """
                You are an expert AI-engineering interviewer reviewing candidate answers.
                Evaluate all answers based on:
                
                **1. Relevance to the question**
                **2. Technical correctness**
                **3. Alignment with the job description (JD)**
                **4. Presence of required keywords (from question metadata)**
                **5. Professionalism and clarity**
                **6. Depth of experience**
                **7. Signal vs noise (usefulness)**
                
                ---------------------------
                ### JOB DESCRIPTION:
                {jd}
                
                ---------------------------
                ### CANDIDATE ANSWERS:
                Provide a structured review for **each answer** below.
                
                Format per answer:
                - **Question ID**
                - **Question Text**
                - **Candidate Answer**
                - **Evaluation** (2–5 sentences)
                - **Score (0–10)** based on relevance + technical depth + JD alignment
                - **Keyword Coverage**: list which required keywords (from question.required_keywords) are present or missing
                
                Finally produce:
                
                ### OVERALL SUMMARY
                - Strengths
                - Weaknesses
                - Hiring Risk Level (Low/Medium/High)
                - Final Overall Score (0–10)
                
                ---------------------------
                ### DATA:
                {answers}
                ---------------------------
                
                Return JSON with this structure:
                
                {{
                  "per_question_reviews": [...],
                  "overall_summary": "...",
                  "final_score": <number 0–10>
                }}
                """
        reviewer_prompt = reviewer_prompt.format(jd=state["jd"], answers=state["answers"])
        response = self.chat_model.invoke([HumanMessage(content=reviewer_prompt)])
        state["review"] = response.content  # attach review to state

        return state

    def node_reporter(self, state: AgentState) -> AgentState:
        reporter_prompt = """
                    You are an expert technical writer.  
                    Your task is to convert the review data into a **clean, polished, executive-quality Markdown report**.
                    
                    The audience is:
                    - Hiring managers
                    - Senior AI/ML engineers
                    - Talent acquisition specialists
                    
                    The tone should be:
                    - Professional
                    - Clear
                    - Concise
                    - Evidence-based
                    
                    ---------------------------
                    ### REVIEW DATA TO SUMMARIZE:
                    
                    {review}
                    
                    ---------------------------
                    
                    ### MARKDOWN REPORT REQUIREMENTS
                    
                    Produce a Markdown document with the following structure:
                    
                    # Candidate Evaluation Report
                    
                    ## 1. Executive Summary
                    - One concise paragraph summarizing overall performance, strengths, and concerns.
                    
                    ## 2. Job Description Alignment
                    Summarize how well the candidate matches the JD requirements.
                    
                    ## 3. Detailed Review by Question
                    For each question:
                    - **Question ID**
                    - **Question Text**
                    - **Score**
                    - **Summary of evaluation**
                    - **Keyword Coverage**
                    - Bullet points highlighting strengths and weaknesses.
                    
                    ## 4. Overall Assessment
                    - Final score (0–10)
                    - Hiring recommendation: **Strong Hire / Hire / Weak Hire / No Hire**
                    
                    ## 5. Risks & Flags
                    Bullet list of any major issues or concerns.
                    
                    Ensure the Markdown is clean and does not include unnecessary JSON dumps.
                    """
                            
        reporter_prompt = reporter_prompt.format(review=state["review"])
        response = self.chat_model.invoke([HumanMessage(content=reporter_prompt)])
        state["report"] = response.content  # attach report to state
        output_path = pathlib.Path("report.md")
        output_path.write_text(state["report"], encoding="utf-8")
        
        print("******")
        pprint(state["report"])
        print("******")
        return state
        
    def router(self, state: AgentState) -> str:
        """Decide the next node based on the current state."""
        if state.get("done"):
            #return "end"
            return "review"
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
    builder.add_node("review", interviewer.node_reviewer)
    builder.add_node("report", interviewer.node_reporter)

    # Entry point is the evaluation node – it will immediately route to “ask”
    builder.set_entry_point("evaluate")

    # After evaluation decide where to go next
    builder.add_conditional_edges(
        "evaluate",
        interviewer.router,
        {
            "ask": "ask",
            #"end": END
            "review": "review"
        },
    )

    # After asking a question we always go back to evaluation (once the UI supplies an answer)
    builder.add_edge("ask", "evaluate")
    builder.add_edge("review", "report")
    builder.add_edge("report", END)
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


