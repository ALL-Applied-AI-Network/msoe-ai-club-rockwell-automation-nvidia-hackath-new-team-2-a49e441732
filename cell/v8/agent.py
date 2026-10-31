"""V8 stage one: a mind that can only talk.

One Pydantic AI agent, NO tools, no senses, no hands - on purpose. Its memory is the message
history, kept whole across the conversation. We judge whether it holds a thought before it is
allowed to see or touch anything.

    uv run python -m v8.agent "hello"      # one exchange from the terminal, for a smoke test
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PERSONA = """You are the mind of a FANUC ER-4iA robot arm in a small work cell at a hackathon,
talking with Joel, who built you. Right now you have NO senses and NO hands: you cannot see the
table, you cannot move anything, and you say so plainly whenever a request would need eyes or arms.
What you can do is talk and remember: hold the thread of the conversation, keep track of what Joel
told you, think out loud with him, and plan. Be direct and brief; no filler, no pretending."""


class Mind:
    """The conversation IS the memory: every exchange carries the full history back in."""

    def __init__(self, model: str | None = None):
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider
        self.model_name = model or os.getenv("V8_MODEL", "openai/gpt-6.1-terra")
        m = OpenAIChatModel(self.model_name,
                            provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]))
        self.agent = Agent(m, instructions=PERSONA)
        self.history: list = []

    def say(self, text: str) -> str:
        r = self.agent.run_sync(text, message_history=self.history)
        self.history = r.all_messages()
        return r.output

    def forget(self) -> None:
        self.history = []


if __name__ == "__main__":
    import sys
    print(Mind().say(sys.argv[1] if len(sys.argv) > 1 else "hello - who are you?"))
