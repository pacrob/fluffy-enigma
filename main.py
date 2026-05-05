"""
Convenience entry point: `uv run python main.py`

The real CLI logic lives in src/chatbot/cli.py.
"""
from chatbot.cli import main

if __name__ == "__main__":
    main()
