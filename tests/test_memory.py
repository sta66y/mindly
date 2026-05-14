from __future__ import annotations

import pytest

from mindly.memory import MemoryLayer


@pytest.fixture
def mem(tmp_path):
    return MemoryLayer(persist_dir=str(tmp_path / "memory"))


def test_store_and_retrieve(mem):
    mem.store("u1", [{"fact": "User wants to lose 10 kg by August", "category": "goal", "do_not_raise": False}])
    results = mem.retrieve("u1", "weight loss goal")
    assert len(results) == 1
    assert "10 kg" in results[0]["fact"]


def test_retrieve_empty_user(mem):
    results = mem.retrieve("nobody", "anything")
    assert results == []


def test_list_all(mem):
    mem.store("u2", [
        {"fact": "Fact A", "category": "goal", "do_not_raise": False},
        {"fact": "Fact B", "category": "event", "do_not_raise": False},
    ])
    all_facts = mem.list_all("u2")
    assert len(all_facts) == 2
    texts = [f["fact"] for f in all_facts]
    assert "Fact A" in texts
    assert "Fact B" in texts


def test_delete_all(mem):
    mem.store("u3", [{"fact": "Will be deleted", "category": "general", "do_not_raise": False}])
    mem.delete("u3", "all")
    assert mem.retrieve("u3", "deleted") == []
    assert mem.list_all("u3") == []


def test_delete_by_query(mem):
    mem.store("u4", [
        {"fact": "User has a dog named Rex", "category": "person", "do_not_raise": False},
        {"fact": "User's goal is meditation", "category": "goal", "do_not_raise": False},
    ])
    mem.delete("u4", "dog Rex")

    remaining = mem.retrieve("u4", "anything", k=10)
    assert not any("Rex" in m["fact"] for m in remaining)


def test_multiple_facts_stored(mem):
    facts = [
        {"fact": f"Fact {i}", "category": "general", "do_not_raise": False}
        for i in range(5)
    ]
    mem.store("u5", facts)
    all_facts = mem.list_all("u5")
    assert len(all_facts) == 5


def test_do_not_raise_roundtrip(mem):
    mem.store("u6", [
        {"fact": "User mentioned divorce", "category": "sensitive", "do_not_raise": True},
    ])
    results = mem.retrieve("u6", "personal life", k=5)
    assert len(results) == 1
    assert results[0]["do_not_raise"] is True
