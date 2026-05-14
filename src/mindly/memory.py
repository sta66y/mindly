from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger


class MemoryLayer:
    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, persist_dir: str = "./data/memory") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=self.EMBED_MODEL,
            device="cpu",
        )
        self._chroma = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
        self._collection = self._chroma.get_or_create_collection(
            name="memories",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        self._db_path = self.persist_dir / "metadata.db"
        self._init_db()
        logger.info(f"MemoryLayer запущен: {self.persist_dir}")

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    fact        TEXT NOT NULL,
                    category    TEXT NOT NULL DEFAULT 'general',
                    do_not_raise INTEGER NOT NULL DEFAULT 0,
                    created_at  REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_uid ON memories(user_id)")
            conn.commit()

    def store(self, user_id: str, facts: list[dict]) -> None:
        if not facts:
            return

        ids, documents, metadatas, db_rows = [], [], [], []
        now = time.time()

        for fact in facts:
            text = (fact.get("fact") or "").strip()
            if not text:
                continue

            fid = str(uuid.uuid4())
            do_not_raise = 1 if fact.get("do_not_raise") else 0
            category = str(fact.get("category", "general"))

            ids.append(fid)
            documents.append(text)
            metadatas.append({
                "user_id": user_id,
                "category": category,
                "do_not_raise": do_not_raise,
                "created_at": now,
            })
            db_rows.append((fid, user_id, text, category, do_not_raise, now))

        if not ids:
            return

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?)",
                db_rows,
            )
            conn.commit()

        logger.info(f"memory.store user={user_id} count={len(ids)}")

    def retrieve(self, user_id: str, query: str, k: int = 7) -> list[dict]:
        total = self._collection.count()
        if total == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, total),
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning(f"memory.retrieve ошибка user={user_id}: {exc}")
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        return [
            {
                "fact": doc,
                "category": meta.get("category", "general"),
                "do_not_raise": bool(meta.get("do_not_raise", 0)),
            }
            for doc, meta in zip(docs, metas)
        ]

    def list_all(self, user_id: str) -> list[dict]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, fact, category, do_not_raise, created_at "
                "FROM memories WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [
            {"id": r[0], "fact": r[1], "category": r[2], "do_not_raise": bool(r[3])}
            for r in rows
        ]

    def delete(self, user_id: str, query: str | Literal["all"]) -> None:
        if query == "all":
            self._delete_all(user_id)
        else:
            self._delete_by_query(user_id, query)

    def _delete_all(self, user_id: str) -> None:
        results = self._collection.get(where={"user_id": user_id})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            conn.commit()
        logger.info(f"memory.delete_all user={user_id} deleted={len(ids)}")

    def _delete_by_query(self, user_id: str, query: str) -> None:
        total = self._collection.count()
        if total == 0:
            return
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(5, total),
                where={"user_id": user_id},
            )
        except Exception as exc:
            logger.warning(f"memory.delete_by_query ошибка: {exc}")
            return

        ids = results.get("ids", [[]])[0]
        if not ids:
            return
        self._collection.delete(ids=ids)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            conn.commit()
        logger.info(f"memory.delete user={user_id} query='{query[:40]}' deleted={len(ids)}")
