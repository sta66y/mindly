from __future__ import annotations

import pytest

from mindly.memory import MemoryLayer


@pytest.fixture
def mem(tmp_path):
    return MemoryLayer(persist_dir=str(tmp_path / "memory"))


def test_isolation_empty_for_new_user(mem):
    mem.store("alice", [{"fact": "Alice has a dog named Biscuit", "category": "person", "do_not_raise": False}])
    bob_mems = mem.retrieve("bob", "dog", k=10)
    assert bob_mems == []


def test_isolation_retrieve_only_own(mem):
    mem.store("alice", [
        {"fact": "Alice loves hiking", "category": "preference", "do_not_raise": False},
        {"fact": "Alice's goal is to run a 5K", "category": "goal", "do_not_raise": False},
    ])
    mem.store("bob", [
        {"fact": "Bob is learning guitar", "category": "goal", "do_not_raise": False},
    ])

    alice_mems = mem.retrieve("alice", "exercise", k=10)
    bob_mems = mem.retrieve("bob", "exercise", k=10)

    alice_texts = [m["fact"] for m in alice_mems]
    bob_texts = [m["fact"] for m in bob_mems]

    assert any("Alice" in t or "hiking" in t or "5K" in t for t in alice_texts)
    assert not any("Alice" in t or "hiking" in t or "5K" in t for t in bob_texts)


def test_isolation_delete_all_scoped(mem):
    mem.store("alice", [{"fact": "Alice fact", "category": "general", "do_not_raise": False}])
    mem.store("bob", [{"fact": "Bob fact", "category": "general", "do_not_raise": False}])

    mem.delete("alice", "all")

    assert mem.list_all("alice") == []
    assert len(mem.list_all("bob")) > 0


def test_no_cross_contamination_after_delete(mem):
    mem.store("alice", [{"fact": "Alice's secret: she hates mornings", "category": "preference", "do_not_raise": True}])
    mem.delete("alice", "all")

    results = mem.retrieve("bob", "hates mornings", k=10)
    assert results == []


def test_do_not_raise_flag_preserved(mem):
    mem.store("alice", [
        {"fact": "Alice mentioned her miscarriage", "category": "sensitive", "do_not_raise": True},
        {"fact": "Alice wants to start journaling", "category": "goal", "do_not_raise": False},
    ])

    mems = mem.retrieve("alice", "health personal", k=10)
    sensitive = [m for m in mems if m["do_not_raise"]]
    regular = [m for m in mems if not m["do_not_raise"]]

    assert len(sensitive) >= 1
    assert any("miscarriage" in m["fact"] for m in sensitive)
    assert len(regular) >= 1
