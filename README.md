Two conversational AI agents (User + Assistant) that talk to each other to book hotels based on live-like data and simulated personas.

- **User Agent**: simulated traveler with Big-5 style personas.
- **Assistant Agent**: two variants
  - Prompt-based assistant (baseline)
  - Fine-tuned assistant (placeholder hook)
- **Retrieval**: hotel candidates loaded via an API abstraction (currently backed by a local JSON file, can later be swapped to TripAdvisor/Yelp/etc.).
- **Memory**: per-session history + optional long-term memory for user preferences.
- **Runner**: scripts to run one or many conversations and log them for evaluation.

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python main.py
