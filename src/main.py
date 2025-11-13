from pathlib import Path

# Import the refactored functions and graph builder
from agents import Interviewer
from typing import List, Dict
import asyncio
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

print(parent_dir)


@dataclass
class JobConfig:
    job_description: str
    questions: List[Dict]


def load_job_config(path: str | Path) -> JobConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return JobConfig(job_description=data["job_description"], questions=data["questions"])


def main() -> None:
    """
    Entry point for the interactive HR interview agent.

    The graph now encapsulates the interview flow:
      * ``evaluate`` is the starting node.
      * ``ask`` asks the main and any follow‑up questions asynchronously.
    This script only loads the configuration, builds the graph, creates the initial
    state, and invokes the graph. All interaction logic lives inside the graph
    nodes.
    """
    # Path to the job configuration JSON file (relative to this script)
    cfg_path = os.path.join(parent_dir, "data/job_config.json")

    # Load job description and interview questions
    config = load_job_config(cfg_path)

    # Build the LangGraph with the Interviewer class nodes
    interviewer_agent = Interviewer()

    result_state = asyncio.run(interviewer_agent.run(config))

    print(result_state)

    # Print a concise summary of the collected answers
    print("\n=== Interview Complete ===\n")
    for ans in result_state.get("answers", []):
        q_id = ans.get("question_id", "N/A")
        question = ans.get("question", "")
        answer = ans.get("answer", "")
        missing = ans.get("missing", [])
        print(f"Q ({q_id}): {question}")
        print(f"A: {answer}")
        if missing:
            print(f"  Missing keywords: {', '.join(missing)}")
        print("-" * 40)


if __name__ == "__main__":
    main()
