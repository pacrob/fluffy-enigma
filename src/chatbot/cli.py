"""
cli.py — interactive command-line interface for the Chatbot.

Run via:
    uv run python -m chatbot.cli
    # or, after `uv pip install -e .`:
    chat

Special commands (prefix with /):
    /quit or /exit  — end the session
    /history        — print the full conversation transcript
    /clear          — erase history and start over
    /help           — show this list

────────────────────────────────────────────────────────────────────────────────
LEARNING NOTE: This file is intentionally thin. All the interesting LangChain
logic lives in bot.py. The CLI's only job is to turn stdin/stdout into calls
to Chatbot.chat(). Keeping I/O separate from business logic makes both easier
to test — the tests in test_bot.py never touch this file.
────────────────────────────────────────────────────────────────────────────────
"""

import sys

from chatbot.bot import Chatbot


HELP_TEXT = """
Commands:
  /help     — show this message
  /history  — print the full conversation so far
  /clear    — erase history and start fresh
  /quit     — exit
  /exit     — exit
""".strip()


def print_history(bot: Chatbot) -> None:
    """Display every message in the conversation history with role labels."""
    messages = bot.get_history()
    if not messages:
        print("  (no history yet)")
        return
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")  # "Human" or "AI"
        print(f"  [{role}] {msg.content}")


def main() -> None:
    """
    Entry point: read from stdin, write to stdout, call bot.chat() in a loop.

    The loop structure illustrates the simplest possible stateful chatbot:
      1. Read user input.
      2. Handle any /commands locally.
      3. Send everything else to the LLM.
      4. Print the response.
      5. Repeat — history accumulates automatically inside `bot`.
    """
    print("Smart CLI Chatbot — Milestone 1")
    print("Type a message and press Enter. Use /help for commands.\n")

    # Chatbot() with no arguments uses ChatOpenAI + OPENAI_API_KEY from .env
    bot = Chatbot()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Gracefully handle Ctrl-D / Ctrl-C
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            command = user_input.lower()

            if command in ("/quit", "/exit"):
                print("Goodbye!")
                sys.exit(0)

            elif command == "/help":
                print(HELP_TEXT)

            elif command == "/history":
                print_history(bot)

            elif command == "/clear":
                bot.clear_history()
                print("  History cleared. Starting fresh.")

            else:
                print(f"  Unknown command: {user_input}. Try /help.")

            continue

        # ── Normal message → LLM ─────────────────────────────────────────────
        try:
            reply = bot.chat(user_input)
            print(f"Bot: {reply}\n")
        except Exception as exc:  # noqa: BLE001
            # Surface API errors without crashing the REPL.
            # Common causes: missing OPENAI_API_KEY, network error, quota exceeded.
            print(f"  Error: {exc}")
            print("  (Check that OPENAI_API_KEY is set in your .env file)\n")


if __name__ == "__main__":
    main()
