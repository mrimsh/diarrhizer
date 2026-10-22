"""Summaries of a finished job's transcript, written by a language model.

Prompt templates (templates.py) and model providers (providers.py) live in
two JSON files shared by the CLI and the GUI; adapters/llm.py talks to the
providers and service.py ties a job's export/result.md to both. Stdlib only:
nothing here may pull in torch/whisperx.
"""
