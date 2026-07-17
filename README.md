# Backend SDE-2 + AI Integration Playground

Welcome to the AI integration playground. This repository contains the code for learning and implementing AI workflows based on the Backend + AI Integration roadmap.

## Project Structure

```
├── .env                       # Local environment configuration (API keys, active model ID)
├── config.py                  # Centralized configuration defaults
├── pyproject.toml             # Standard setuptools configuration for package management
├── requirements.txt           # LiteLLM and production dependencies
├── NOTES.md                   # Interview recall notes, tradeoffs, and gotchas
│
└── day01/                     # Day 1: LLM APIs Playground (Using LiteLLM)
    ├── 01_basic_generation.py # Text generation & streaming response examples
    ├── 02_structured_output.py# Pydantic schema validation & JSON extraction
    ├── 03_function_calling.py # Declarative tool registration & agent loop execution
    ├── 04_token_counter.py    # Pre-flight token counting & dynamic cost tracking
    └── data/                  # Extracted mock databases & sample logs
```

## Getting Started

1. **Install dependencies and the project package in editable mode globally**:
   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```
2. **Configure API Keys**:
   Set `MODEL_ID` and keys inside the `.env` file at the root.

## Swapping Providers (Plug-and-Play)

Swapping providers is zero-code. To switch from Gemini to OpenAI:
1. Update `.env`:
   ```env
   MODEL_ID=openai/gpt-4o
   OPENAI_API_KEY=sk-...
   ```
No day-specific playground scripts or setup configurations need to change.

## Running scripts

Run scripts from the project root using standard python paths:
```powershell
python day01/01_basic_generation.py
python day01/02_structured_output.py
python day01/03_function_calling.py
python day01/04_token_counter.py
```
*(All imports resolve cleanly via the local package module structure)*
