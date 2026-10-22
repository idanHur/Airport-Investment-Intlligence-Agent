# Airport Investment Intelligence Agent

See [ASSIGNMENT.md](ASSIGNMENT.md) for the task.

## Setup (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # then put your OPENAI_API_KEY in .env
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
