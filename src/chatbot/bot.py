"""
bot.py — the core LangChain chatbot.

────────────────────────────────────────────────────────────────────────────────
LEARNING GUIDE: Key LangChain Concepts Used Here
────────────────────────────────────────────────────────────────────────────────

1. ChatOpenAI (the LLM wrapper)
   ─────────────────────────────
   LangChain wraps every language model behind a common interface. ChatOpenAI
   speaks to OpenAI's chat-completion endpoint. Because all LLM wrappers share
   the same interface, you can swap providers (Anthropic, Google, local Ollama)
   by changing one import — none of the surrounding pipeline code changes.

   Key idea: LangChain separates *what model to use* from *how to use it*.

2. ChatPromptTemplate (structured prompt authoring)
   ──────────────────────────────────────────────────
   A template turns abstract "slots" into a concrete list of chat messages.
   It lets you describe the conversation structure declaratively:

       [("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),   ← conversation so far goes here
        ("human", "{input}")]             ← current user turn goes here

   At call time, .invoke({"input": "...", "history": [...]}) fills in the
   slots and produces the final prompt that gets sent to the model.

   Key idea: keep *structure* (the template) separate from *content* (the data).

3. InMemoryChatMessageHistory (the memory store)
   ────────────────────────────────────────────────
   This is a simple in-memory list of LangChain message objects
   (HumanMessage, AIMessage, SystemMessage). Every time the chain runs,
   the old messages are read from this store and injected into the prompt via
   the MessagesPlaceholder above, giving the model its "memory".

   In a real app you'd swap this for a database-backed store
   (e.g. RedisChatMessageHistory) without changing any other code.

   Key idea: the store is pluggable — the chain doesn't care where history comes from.

4. LCEL pipe operator  prompt | llm
   ──────────────────────────────────
   The `|` operator composes Runnables into a chain (LangChain Expression
   Language, LCEL). Each step's output becomes the next step's input:

       prompt  →  formats messages
       llm     →  generates an AIMessage response

   You can keep chaining:  prompt | llm | StrOutputParser() | ...

5. RunnableWithMessageHistory (automatic history injection)
   ──────────────────────────────────────────────────────────
   This wraps any chain and handles the history bookkeeping automatically:
     • Before the chain runs: loads messages from the store → fills MessagesPlaceholder.
     • After the chain runs:  saves the new human + AI messages back to the store.

   You just call .invoke() normally; history is handled transparently.

────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Load .env so OPENAI_API_KEY is available when running outside tests.
load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer the user's questions clearly. "
    "If you don't know something, say so."
)


class Chatbot:
    """
    A stateful conversational chatbot built on LangChain primitives.

    Usage
    ─────
        bot = Chatbot()          # uses ChatOpenAI + OPENAI_API_KEY from env
        reply = bot.chat("Hi!")  # returns str
        history = bot.get_history()
        bot.clear_history()

    Dependency injection (for tests / swapping providers)
    ──────────────────────────────────────────────────────
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        fake_llm = FakeListChatModel(responses=["Ahoy!"])
        bot = Chatbot(llm=fake_llm, system_prompt="You are a pirate.")
    """

    def __init__(
        self,
        llm: Any | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        """
        Build the chain and attach the history store.

        Parameters
        ──────────
        llm
            Any LangChain chat model (ChatOpenAI, FakeListChatModel, …).
            When None, a ChatOpenAI instance is created using OPENAI_API_KEY
            from the environment.
        system_prompt
            The text placed in the "system" role at the top of every request.
            Use it to set the model's persona, rules, or context.
        model_name
            OpenAI model identifier — only used when `llm` is None.
        """
        # ── 1. The LLM ────────────────────────────────────────────────────────
        # Accept an injected model (useful for testing) or build a real one.
        # ChatOpenAI reads OPENAI_API_KEY from the environment automatically.
        self._llm = llm if llm is not None else ChatOpenAI(model=model_name)

        # ── 2. The prompt template ────────────────────────────────────────────
        # ChatPromptTemplate.from_messages() accepts a list of (role, content)
        # pairs plus special placeholders.
        #
        # MessagesPlaceholder("history") is a slot that expands into however
        # many messages are in the history store — zero on the first turn,
        # growing with each exchange.
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        # ── 3. The base chain (LCEL pipe) ─────────────────────────────────────
        # `prompt | llm` creates a Runnable that:
        #   a) calls prompt.invoke(data)  → list[BaseMessage]
        #   b) calls llm.invoke(messages) → AIMessage
        self._chain = self._prompt | self._llm

        # ── 4. The history store ──────────────────────────────────────────────
        # InMemoryChatMessageHistory is a plain Python list dressed as a
        # LangChain history store. Clear it → start a new conversation.
        self._history = InMemoryChatMessageHistory()

        # ── 5. Wrap the chain with automatic history management ───────────────
        # RunnableWithMessageHistory intercepts every .invoke() call:
        #   BEFORE: reads self._history and injects it as "history"
        #   AFTER:  appends the new human + AI messages to self._history
        #
        # The lambda `lambda session_id: self._history` is a factory function.
        # In multi-user apps you'd look up a per-user store by session_id.
        # Here we always return the same single store.
        self._chain_with_history = RunnableWithMessageHistory(
            self._chain,
            lambda session_id: self._history,
            input_messages_key="input",      # the key in our invoke() dict that holds the human message
            history_messages_key="history",  # must match the MessagesPlaceholder variable_name above
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        """
        Send `message` to the LLM and return the reply as a plain string.

        The full conversation history is automatically included in the prompt
        so the model has context from earlier turns.

        Parameters
        ──────────
        message : str
            The user's next message.

        Returns
        ───────
        str
            The model's reply (AIMessage.content unwrapped to a plain str).
        """
        # RunnableWithMessageHistory.invoke() needs a session identifier so it
        # knows which history store to use. We use a fixed key because this
        # Chatbot instance manages exactly one conversation.
        response = self._chain_with_history.invoke(
            {"input": message},
            config={"configurable": {"session_id": "default"}},
        )
        # response is an AIMessage; .content is the plain text string.
        return response.content

    def get_history(self) -> list[BaseMessage]:
        """
        Return the full conversation history as a list of LangChain messages.

        Each element is either a HumanMessage or an AIMessage. They alternate
        in chronological order: [Human, AI, Human, AI, …]

        Useful for: inspecting context, displaying a transcript, persisting
        the conversation, or feeding history to another chain.
        """
        return self._history.messages

    def clear_history(self) -> None:
        """
        Erase all conversation history and start fresh.

        The next call to chat() will behave as if this is the first message —
        the model will have no memory of prior exchanges.
        """
        self._history.clear()
