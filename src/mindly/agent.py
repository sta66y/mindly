from __future__ import annotations

import json
import threading
import time
from typing import Iterator, Literal

from loguru import logger
from openai import OpenAI

from .memory import MemoryLayer
from .personas import PERSONAS

_FACT_EXTRACTION_PROMPT = """\
Extract factual information about the USER from this conversation excerpt.
Focus on: personal goals, family members, life events, preferences, recurring patterns, \
and any topic the user explicitly asked NOT to bring up again.

Return a JSON array of objects. Each object must have:
  "fact"         : string — the fact, in plain language
  "category"     : one of [goal, person, preference, event, pattern, sensitive]
  "do_not_raise" : boolean — true ONLY if the user explicitly said not to raise this topic

Rules:
- Extract facts about the USER only, not the assistant.
- If the user says "don't bring that up" or "let's not talk about X", set do_not_raise=true for that fact.
- Return [] if there are no new facts worth storing.
- Return ONLY the raw JSON array. No markdown fences. No explanation.

Conversation:
{conversation}"""


class MindlyAgent:
    def __init__(
        self,
        memory: MemoryLayer,
        llm_client: OpenAI,
        model: str,
        memory_k: int = 7,
        history_turns: int = 6,
    ) -> None:
        self.memory = memory
        self.client = llm_client
        self.model = model
        self.memory_k = memory_k
        self.history_turns = history_turns
        self._sessions: dict[str, list[dict]] = {}
        self._bg_threads: list[threading.Thread] = []

    def chat(
        self,
        user_id: str,
        persona: str,
        message: str,
        stream: bool = True,
    ) -> Iterator[str] | str:
        t_start = time.perf_counter()

        memories = self.memory.retrieve(user_id, message, k=self.memory_k)
        system_prompt = self._build_system_prompt(persona, memories)

        history = self._sessions.get(user_id, [])
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history[-self.history_turns * 2:]
            + [{"role": "user", "content": message}]
        )

        logger.info(
            f"chat user={user_id} persona={persona} "
            f"memories={len(memories)} history_turns={len(history)//2}"
        )

        if stream:
            return self._stream(user_id, persona, message, messages, t_start)
        else:
            return self._generate(user_id, persona, message, messages, t_start)

    def forget(self, user_id: str, query: str | Literal["all"]) -> None:
        self.memory.delete(user_id, query)
        if query == "all":
            self._sessions.pop(user_id, None)
        logger.info(f"forget user={user_id} query='{query}'")

    def new_session(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    def ingest_conversation(self, user_id: str, turns: list[dict]) -> None:
        self._extract_and_store(user_id, turns)

    def flush_background(self, timeout: float = 30.0) -> None:
        for t in self._bg_threads:
            t.join(timeout=timeout)
        self._bg_threads.clear()

    def _generate(
        self,
        user_id: str,
        persona: str,
        message: str,
        messages: list[dict],
        t_start: float,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        text = resp.choices[0].message.content or ""
        ttft = time.perf_counter() - t_start
        logger.info(f"chat.generate user={user_id} TTFT={ttft:.3f}s tokens={len(text.split())}")
        self._post_turn(user_id, message, text)
        return text

    def _stream(
        self,
        user_id: str,
        persona: str,
        message: str,
        messages: list[dict],
        t_start: float,
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        chunks: list[str] = []
        first_token = True

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token:
                    ttft = time.perf_counter() - t_start
                    logger.info(f"chat.stream user={user_id} TTFT={ttft:.3f}s")
                    first_token = False
                chunks.append(delta)
                yield delta

        full_text = "".join(chunks)
        self._post_turn(user_id, message, full_text)

    def _post_turn(self, user_id: str, user_msg: str, agent_msg: str) -> None:
        history = self._sessions.setdefault(user_id, [])
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": agent_msg})

        turns = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": agent_msg},
        ]
        t = threading.Thread(
            target=self._extract_and_store,
            args=(user_id, turns),
            daemon=True,
        )
        t.start()
        self._bg_threads.append(t)

    def _extract_and_store(self, user_id: str, turns: list[dict]) -> None:
        facts = self._extract_facts(user_id, turns)
        if facts:
            self.memory.store(user_id, facts)

    def _extract_facts(self, user_id: str, turns: list[dict]) -> list[dict]:
        conversation_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns
        )
        prompt = _FACT_EXTRACTION_PROMPT.format(conversation=conversation_text)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                stream=False,
            )
            raw = (resp.choices[0].message.content or "").strip()

            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            facts = json.loads(raw)
            if not isinstance(facts, list):
                return []
            logger.debug(f"facts_extracted user={user_id} count={len(facts)}")
            return facts
        except Exception as exc:
            logger.error(f"fact_extraction_failed user={user_id}: {exc}")
            return []

    def _build_system_prompt(self, persona: str, memories: list[dict]) -> str:
        base = PERSONAS.get(persona, PERSONAS["wellness_friend"])["system_prompt"]

        regular = [m for m in memories if not m["do_not_raise"]]
        sensitive = [m for m in memories if m["do_not_raise"]]

        blocks: list[str] = [base]

        if regular:
            facts_md = "\n".join(f"- {m['fact']}" for m in regular)
            blocks.append(f"\n## What you know about this client:\n{facts_md}")

        if sensitive:
            facts_md = "\n".join(f"- {m['fact']}" for m in sensitive)
            blocks.append(
                f"\n## Topics to NEVER raise proactively:\n{facts_md}\n"
                "(You may acknowledge these topics only if the client explicitly brings them up first.)"
            )

        if regular or sensitive:
            blocks.append(
                "\nUse this knowledge naturally — reference it when relevant, "
                "ask follow-up questions to show you remember."
            )

        return "\n".join(blocks)
