"""
Tests for Chatbot — written BEFORE the implementation (Red phase of TDD).

Why FakeListChatModel?
  LangChain ships a fake LLM that returns pre-scripted strings without any
  network call. Using it here means:
    - Tests run offline (no OpenAI key required).
    - The full LangChain pipeline (prompt formatting, history management) is
      exercised for real — only the HTTP round-trip is replaced.
  This is better than a raw MagicMock because it validates the shape of the
  data flowing through the chain.

Run with:
    uv run pytest -v
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

# ── This import will FAIL until we create src/chatbot/bot.py ──────────────────
from chatbot import Chatbot


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_fake_llm(*responses: str) -> FakeListChatModel:
    """
    Build a FakeListChatModel that returns `responses` in order.

    FakeListChatModel accepts a list of plain strings; it wraps each one in
    an AIMessage automatically, which is exactly what a real ChatOpenAI returns.
    """
    return FakeListChatModel(responses=list(responses))


@pytest.fixture
def bot_one_shot():
    """A Chatbot wired to return exactly one pre-scripted reply."""
    llm = make_fake_llm("Hello! How can I help you?")
    return Chatbot(llm=llm)


@pytest.fixture
def bot_multi_turn():
    """A Chatbot wired to return three replies in sequence."""
    llm = make_fake_llm("I'm doing great!", "Paris is the capital.", "Goodbye!")
    return Chatbot(llm=llm)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInstantiation:
    def test_chatbot_accepts_injected_llm(self):
        """Chatbot.__init__ must accept an `llm` keyword argument."""
        llm = make_fake_llm("hi")
        bot = Chatbot(llm=llm)
        assert bot is not None

    def test_chatbot_accepts_custom_system_prompt(self):
        """The system prompt should be configurable at construction time."""
        llm = make_fake_llm("hi")
        bot = Chatbot(llm=llm, system_prompt="You are a pirate. Speak like one.")
        assert bot is not None


class TestHistory:
    def test_history_starts_empty(self, bot_one_shot):
        """A freshly created Chatbot has no messages in its history."""
        assert bot_one_shot.get_history() == []

    def test_history_records_human_and_ai_messages(self, bot_one_shot):
        """After one chat() call, history must contain one HumanMessage and one AIMessage."""
        bot_one_shot.chat("Hi there")
        history = bot_one_shot.get_history()

        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)

    def test_history_captures_correct_content(self, bot_one_shot):
        """The content of the stored messages must match what was sent/received."""
        bot_one_shot.chat("Hi there")
        history = bot_one_shot.get_history()

        assert history[0].content == "Hi there"
        assert history[1].content == "Hello! How can I help you?"

    def test_history_grows_across_multiple_turns(self, bot_multi_turn):
        """Each turn appends two messages (human + AI); history grows linearly."""
        bot_multi_turn.chat("How are you?")
        bot_multi_turn.chat("What is the capital of France?")

        history = bot_multi_turn.get_history()
        assert len(history) == 4  # 2 turns × 2 messages each

    def test_history_preserves_alternating_roles(self, bot_multi_turn):
        """Messages must alternate Human→AI→Human→AI in the stored history."""
        bot_multi_turn.chat("How are you?")
        bot_multi_turn.chat("What is the capital of France?")

        history = bot_multi_turn.get_history()
        roles = [type(m).__name__ for m in history]
        assert roles == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]

    def test_clear_history_empties_the_list(self, bot_multi_turn):
        """clear_history() must remove all messages."""
        bot_multi_turn.chat("First message")
        bot_multi_turn.clear_history()

        assert bot_multi_turn.get_history() == []

    def test_chat_works_after_clear(self, bot_multi_turn):
        """clear_history() must not break subsequent conversations."""
        bot_multi_turn.chat("First message")
        bot_multi_turn.clear_history()
        bot_multi_turn.chat("Second message")

        history = bot_multi_turn.get_history()
        assert len(history) == 2
        assert history[0].content == "Second message"


class TestChatResponse:
    def test_chat_returns_a_string(self, bot_one_shot):
        """chat() must return a plain Python str, not a LangChain message object."""
        result = bot_one_shot.chat("Hello")
        assert isinstance(result, str)

    def test_chat_returns_correct_text(self, bot_one_shot):
        """The returned string must match the LLM's scripted reply."""
        result = bot_one_shot.chat("Hello")
        assert result == "Hello! How can I help you?"

    def test_multiple_calls_return_different_replies(self, bot_multi_turn):
        """Successive calls consume successive scripted responses in order."""
        r1 = bot_multi_turn.chat("How are you?")
        r2 = bot_multi_turn.chat("What is the capital of France?")

        assert r1 == "I'm doing great!"
        assert r2 == "Paris is the capital."
