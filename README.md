# Airport Investment Intelligence Agent

See [ASSIGNMENT.md](ASSIGNMENT.md) for the task.

## Setup (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # then put your OPENAI_API_KEY in .env
```

## Run

```powershell
.\.venv\Scripts\python -m airport_agent.data.build_cache   # one-time: downloads and caches all data (slow on first run)
.\.venv\Scripts\streamlit run app.py                       # chat UI
.\.venv\Scripts\python -m airport_agent.agent              # or: terminal chat
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
