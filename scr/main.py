from pathlib import Path

# Import the refactored functions and graph builder
from agents import load_job_config, build_graph, initial_state_from_config

import asyncio

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
    cfg_path = Path(__file__).parent / "data" / "job_config.json"

    # Load job description and interview questions
    config = load_job_config(cfg_path)

    # Build the LangGraph with the Interviewer class nodes
    graph = build_graph(config)

    # Initialise the LangGraph state from the configuration
    state = initial_state_from_config(config)

    # Async runner that uses the graph's async invoke method.
    async def run_async():
        # ``evaluate`` is the entry point; the graph will handle async ask node.
        result_state = await graph.ainvoke(state, {"configurable": {"thread_id": "ui", "recursion_limit": 100}})
        return result_state

    result_state = asyncio.run(run_async())

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
