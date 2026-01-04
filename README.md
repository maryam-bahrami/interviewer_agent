# HR Interviewer Agent

## Overview

The **HR Interviewer Agent** is a LangChain/LangGraph‑based AI interview system that automates the end‑to‑end hiring questionnaire workflow. It:

- Loads a job description and a list of interview questions from a configurable JSON file.
- Interactively asks the candidate each question via a Gradio web UI.
- Evaluates answers in real time using a large language model (LLM) and generates follow‑up questions when needed.
- Produces a structured review of the candidate’s responses and a polished Markdown report.

The project demonstrates how to combine **LangGraph** state graphs, **LangChain** LLM wrappers, and **Gradio** for a responsive, asynchronous chat‑style interview experience.

## Architecture

```
└─ scr/
   ├─ app.py          # Gradio UI + session management
   ├─ agents.py       # Core LangGraph graph, Interviewer class, state definitions
   └─ requirements.txt
```

- **`agents.py`** defines the `JobConfig` dataclass, the `AgentState` TypedDict, and the `Interviewer` class that contains the LLM logic for asking questions, evaluating answers, reviewing the interview, and generating a final report.  
- **`app.py`** creates a per‑session graph, handles asynchronous user input, and renders the chat UI with Gradio.  
- **`requirements.txt`** lists the Python dependencies required to run the system.

## Setup

### Prerequisites

- Python 3.9 or newer
- Access to an LLM endpoint (OpenAI, Azure, etc.) – the default model is `gpt-4o-mini`
- `git` (optional, for cloning the repository)

### Installation

```bash
# Clone the repository (if not already)
git clone <repo‑url>
cd <the_project_folder>

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r scr/requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (or set the variables in your shell) with the following keys:

| Variable                | Description                                            |
|-------------------------|--------------------------------------------------------|
| `JOB_DESCRIPTION_PATH`  | Path to a JSON file containing the job description and interview questions. |
| `MODEL_NAME`            | Name of the LLM model to use (default: `gpt-4o-mini`). |
| `BASE_URL`              | Base URL of the LLM API (optional, required for non‑OpenAI providers). |
| `API_KEY`               | API key for the LLM service (default placeholder `not-needed`). |

Example `.env`:

```dotenv
JOB_DESCRIPTION_PATH=./job_config.json
MODEL_NAME=gpt-4o-mini
BASE_URL=https://api.openai.com/v1
API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

### Job Description JSON

The JSON file referenced by `JOB_DESCRIPTION_PATH` must follow this schema:

```json
{
  "job_description": "Full text describing the role.",
  "questions": [
    {
      "id": "q1",
      "text": "First interview question?",
      "required_keywords": ["keyword1", "keyword2"],
      "guidance": "Optional guidance for the candidate."
    }
    // … more questions …
  ],
  "number_of_followup_chances": 2
}
```

- `required_keywords` are used by the LLM evaluator to decide whether a follow‑up is needed.
- `number_of_followup_chances` limits how many follow‑up questions can be asked per original question.

## Running the Interview Bot

```bash
python scr/app.py
```

The Gradio UI will start on `http://0.0.0.0:7860`. Click **Start Interview** to begin. The system will:

1. Load the job configuration.
2. Prompt the candidate with the first question.
3. Evaluate the answer, optionally ask follow‑up questions, and move to the next question.
4. After all questions are answered, generate a review and a Markdown report (`report.md`) in the working directory.

## Output

- **`report.md`** – a polished Markdown document containing per‑question reviews, an overall summary, and a final score.
- Console logs provide debugging information and the raw LLM JSON responses.

## Extending the Project

- **Custom LLMs** – modify `Interviewer.__init__` to use a different provider or model.
- **Additional Nodes** – add new LangGraph nodes (e.g., sentiment analysis, plagiarism check) and update the conditional routing in `build_graph`.
- **Front‑end Enhancements** – replace Gradio with Streamlit, FastAPI, or a custom React front‑end while keeping the same graph logic.

## License

This project is provided under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- **LangChain** – for the LLM abstraction layer.
- **LangGraph** – for the state‑graph workflow engine.
- **Gradio** – for the rapid UI prototyping.

--- 

*For any issues or contributions, please open a GitHub issue or submit a pull request.*
