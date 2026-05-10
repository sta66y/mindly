from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from mindly.agent import MindlyAgent
from mindly.memory import MemoryLayer

load_dotenv()

DEMO_MEMORY_DIR = "./data/demo_isolation_memory"


def make_agent() -> MindlyAgent:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    memory = MemoryLayer(persist_dir=DEMO_MEMORY_DIR)
    return MindlyAgent(
        memory=memory,
        llm_client=client,
        model=os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct:free"),
    )


def banner(text: str) -> None:
    print(f"\n{'='*60}\n{text}\n{'='*60}")


def main() -> None:
    agent = make_agent()

    agent.forget("alice", "all")
    agent.forget("bob", "all")

    banner("СЕССИЯ 1 — Алиса рассказывает о себе")
    alice_msg = (
        "Hi! I'm Alice. My son Max is in 9th grade and really stressed about exams. "
        "I also just started running — did 3km today for the first time ever!"
    )
    print(f"Алиса: {alice_msg}")
    response = agent.chat("alice", "wellness_friend", alice_msg, stream=False)
    print(f"Агент → Алиса: {response[:200]}...")
    agent.flush_background(timeout=30)
    print(f"\nСохранено фактов об Алисе: {len(agent.memory.list_all('alice'))}")

    banner("СЕССИЯ 2 — Боб начинает с чистого листа")
    bob_msg = "Hey, what do you know about me?"
    print(f"Боб: {bob_msg}")
    bob_response = agent.chat("bob", "wellness_friend", bob_msg, stream=False)
    print(f"Агент → Боб: {bob_response}")

    print("\n--- Проверка изоляции ---")
    bob_memories = agent.memory.list_all("bob")
    alice_memories = agent.memory.list_all("alice")

    print(f"Фактов об Алисе: {len(alice_memories)}")
    print(f"Фактов о Бобе:   {len(bob_memories)}")

    leaked = any(
        "Alice" in m["fact"] or "Max" in m["fact"] or "9th grade" in m["fact"]
        for m in bob_memories
    )
    assert not leaked, "ПРОВАЛ: факты Алисы утекли в память Боба!"

    response_leaked = "Max" in bob_response or "9th grade" in bob_response or "Alice" in bob_response
    assert not response_leaked, "ПРОВАЛ: факты Алисы появились в ответе Бобу!"

    print("✅ Tenant isolation: PASSED — Боб ничего не знает об Алисе")

    agent.forget("alice", "all")
    agent.forget("bob", "all")
    import shutil
    shutil.rmtree(DEMO_MEMORY_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
