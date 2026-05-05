# Milestone 1 — Hello, LLM

A command-line chatbot that maintains conversation history, built to learn
the core LangChain primitives:

| Concept | LangChain class |
|---|---|
| LLM wrapper | `ChatOpenAI` |
| Prompt authoring | `ChatPromptTemplate` + `MessagesPlaceholder` |
| Conversation memory | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` |
| Composing steps | LCEL `\|` operator |

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd fluffy-enigma

# 2. Copy the env template and add your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Run the chatbot
uv run python main.py
```

```
Smart CLI Chatbot — Milestone 1
Type a message and press Enter. Use /help for commands.

You: Who wrote Hamlet?
Bot: William Shakespeare wrote Hamlet, likely around 1600–1601.

You: What else did he write?
Bot: Shakespeare wrote 37+ plays including Macbeth, Othello, King Lear...
```

The second answer works because the model sees the full conversation history —
that is the key thing this milestone is teaching.

---

## Project Layout

```
fluffy-enigma/
├── src/
│   └── chatbot/
│       ├── __init__.py   # public surface: `from chatbot import Chatbot`
│       ├── bot.py        # ← all the LangChain logic lives here
│       └── cli.py        # thin stdin/stdout loop; calls bot.chat()
├── tests/
│   └── test_bot.py       # 12 tests; run with `uv run pytest -v`
├── main.py               # `uv run python main.py` entry point
├── pyproject.toml
└── .env.example
```

---

## How It Works — A Deep Dive

### 1. The LLM Wrapper: `ChatOpenAI`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

LangChain wraps every language model behind a **common interface** (`BaseChatModel`).
`ChatOpenAI` is one implementation; others include `ChatAnthropic`, `ChatGoogleGenerativeAI`,
`ChatOllama` (local models), etc.

Because they all share the same interface you can swap providers by changing
**one import** — none of the surrounding chain code changes:

```python
# Switch from OpenAI to a local Ollama model:
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")
# Everything else stays the same.
```

**What `ChatOpenAI.invoke()` actually does:**
1. Accepts a list of LangChain message objects (`HumanMessage`, `AIMessage`, `SystemMessage`).
2. Serialises them into the format OpenAI's API expects.
3. Makes an HTTP POST to `https://api.openai.com/v1/chat/completions`.
4. Deserialises the JSON response into a LangChain `AIMessage`.

You rarely call `.invoke()` on the model directly; instead you compose it into
a chain (see LCEL below).

---

### 2. Structuring Prompts: `ChatPromptTemplate`

Modern LLMs expect a **list of messages**, not a flat string. Each message has
a role (`system`, `human`, `ai`) that tells the model who said it.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, concise assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← expands to N messages
    ("human", "{input}"),                          # ← current user turn
])
```

**What the three parts do:**

| Part | Role | Purpose |
|---|---|---|
| `("system", "...")` | `system` | Persistent instructions; the model's persona |
| `MessagesPlaceholder("history")` | mixed | All prior messages expand here; zero on turn 1 |
| `("human", "{input}")` | `human` | The user's current message |

**Filling the template:**

```python
# prompt.invoke() returns a list of chat messages ready for the LLM.
messages = prompt.invoke({
    "input": "Who wrote Hamlet?",
    "history": [],  # empty on the first turn
})
# → [SystemMessage("You are a helpful..."),
#    HumanMessage("Who wrote Hamlet?")]
```

On turn 2 the history list already has two messages in it, so the model sees
the full conversation context.

**Key idea:** A template separates *structure* (which roles, which slots) from
*content* (the actual text filled in at runtime). You define the structure once
and reuse it for every message.

---

### 3. Conversation Memory: `InMemoryChatMessageHistory`

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
```

This is the simplest possible memory store: a plain Python list dressed up as
a LangChain interface. It holds `BaseMessage` objects in chronological order:

```
[HumanMessage("Who wrote Hamlet?"),
 AIMessage("William Shakespeare..."),
 HumanMessage("What else did he write?"),
 AIMessage("Shakespeare wrote 37+ plays...")]
```

**Why is it a separate object?**
Because the *store* is pluggable. You can swap `InMemoryChatMessageHistory`
for a database-backed store without changing any chain code:

```python
# Persist to Redis across restarts:
from langchain_community.chat_message_histories import RedisChatMessageHistory
history = RedisChatMessageHistory(session_id="user-123", url="redis://localhost")

# Or to a SQLite database:
from langchain_community.chat_message_histories import SQLChatMessageHistory
history = SQLChatMessageHistory(session_id="user-123", connection="sqlite:///chat.db")
```

**`ConversationBufferMemory` — the classic approach**

Older LangChain tutorials (pre-1.x) show `ConversationBufferMemory`. It was
removed in LangChain 1.x in favour of the pattern shown here (explicit history
store + `RunnableWithMessageHistory`). The underlying concept is identical:
buffer every message, inject the whole buffer into each prompt.

---

### 4. Composing Steps: LCEL (LangChain Expression Language)

```python
chain = prompt | llm
```

The `|` operator wires LangChain `Runnable` objects into a **pipeline**.
Every step's output becomes the next step's input:

```
{"input": "Hi", "history": [...]}
        ↓  prompt
[SystemMessage, HumanMessage("Hi")]
        ↓  llm
AIMessage("Hello! How can I help you?")
```

You can keep adding steps:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
# now chain.invoke(...) returns a plain str instead of AIMessage
```

**Why LCEL?**
- **Composability:** snap pieces together like LEGO.
- **Streaming:** every step supports `.stream()` with no extra code.
- **Observability:** LangSmith traces every step automatically.
- **Parallelism:** steps on separate branches run concurrently.

---

### 5. Automatic History Management: `RunnableWithMessageHistory`

Manually reading history, passing it to the chain, and saving the result
would be tedious. `RunnableWithMessageHistory` automates all three:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,                              # the base chain to wrap
    lambda session_id: history,         # returns the store for this session
    input_messages_key="input",         # key in invoke() dict = current human turn
    history_messages_key="history",     # MessagesPlaceholder name in the prompt
)
```

**What happens on every `chain_with_history.invoke()` call:**

```
invoke({"input": "What else did he write?"}, config={"configurable": {"session_id": "..."}})
    │
    ├─ BEFORE chain runs:
    │    reads history store → injects messages into "history" slot
    │
    ├─ chain runs:
    │    prompt formats messages → llm generates response
    │
    └─ AFTER chain runs:
         saves HumanMessage("What else did he write?") to store
         saves AIMessage("Shakespeare wrote 37+ plays...") to store
```

**The `session_id`** lets one `RunnableWithMessageHistory` serve multiple
concurrent conversations by calling the factory lambda with different IDs.
In this project every `Chatbot` instance has exactly one conversation, so
we always pass `"default"`.

**Deprecation note:** LangChain 1.x marks `RunnableWithMessageHistory` as
deprecated in favour of LangGraph's built-in persistence. For this milestone
it remains the clearest illustration of the concept. Milestone 2+ will
introduce LangGraph.

---

### Full Data Flow Diagram

```
User types "What else did he write?"
            │
            ▼
        bot.chat("What else did he write?")
            │
            ▼
  chain_with_history.invoke(
      {"input": "What else did he write?"},
      config={"configurable": {"session_id": "default"}}
  )
            │
            ├─ reads history:
            │    [HumanMessage("Who wrote Hamlet?"),
            │     AIMessage("William Shakespeare...")]
            │
            ▼
  prompt.invoke({
      "input": "What else did he write?",
      "history": [HumanMessage(...), AIMessage(...)]
  })
            │
            ▼
  [SystemMessage("You are a helpful assistant."),
   HumanMessage("Who wrote Hamlet?"),       ← from history
   AIMessage("William Shakespeare..."),     ← from history
   HumanMessage("What else did he write?")] ← current turn
            │
            ▼
  llm.invoke([...messages...])   → HTTP POST to OpenAI
            │
            ▼
  AIMessage("Shakespeare wrote 37+ plays...")
            │
            ├─ saves to history:
            │    HumanMessage("What else did he write?")
            │    AIMessage("Shakespeare wrote 37+ plays...")
            │
            ▼
  returns "Shakespeare wrote 37+ plays..."   (str)
```

---

## Running the Tests

Tests use `FakeListChatModel` — a LangChain-provided fake that returns
pre-scripted strings without any network call. All 12 tests run offline:

```bash
uv run pytest -v
```

**Why `FakeListChatModel` instead of `unittest.mock`?**

`FakeListChatModel` is a *real* `BaseChatModel` implementation. Using it means
the full LangChain pipeline — prompt formatting, LCEL composition, history
injection — runs exactly as it would in production. Only the HTTP call is
replaced. A raw `MagicMock` would bypass most of the code under test.

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

fake_llm = FakeListChatModel(responses=["Hello!", "Goodbye!"])
bot = Chatbot(llm=fake_llm)

assert bot.chat("Hi") == "Hello!"
assert bot.chat("Bye") == "Goodbye!"
```

---

## CLI Commands

| Command | Effect |
|---|---|
| (any text) | Send to the LLM; print the reply |
| `/help` | Show available commands |
| `/history` | Print the full conversation so far |
| `/clear` | Erase history and start a new conversation |
| `/quit` or `/exit` | Exit the program |
| `Ctrl-C` / `Ctrl-D` | Exit gracefully |

---

## Key Takeaways from Milestone 1

1. **LangChain wraps LLMs behind a common interface** — swap providers by
   changing one import.

2. **`ChatPromptTemplate` separates structure from content** — define the
   conversation shape once; fill in slots at runtime.

3. **Memory is just a list** — `InMemoryChatMessageHistory` is a list of
   message objects injected into the prompt before each LLM call. "Memory"
   is not magic; it's the model re-reading the transcript.

4. **LCEL `|` composes Runnables** — `prompt | llm` is a reusable,
   streamable, observable pipeline.

5. **`RunnableWithMessageHistory` automates bookkeeping** — read history →
   run chain → write history, all transparent to the caller.

---

## What's Next (Milestone 2)

- **Tools** — give the model the ability to call Python functions (search,
  calculator, database queries).
- **LangGraph** — replace `RunnableWithMessageHistory` with proper stateful
  graphs and built-in persistence.
- **Streaming** — print tokens as they arrive instead of waiting for the
  full response.
