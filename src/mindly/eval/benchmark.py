from __future__ import annotations

import os
import time
from typing import Optional

from loguru import logger

from ..agent import MindlyAgent


def _substring_score(expected: str, predicted: str) -> float:
    return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0


def run_longmemeval(
    agent: MindlyAgent,
    n_samples: int = 100,
    split: str = "oracle",
    wandb_project: Optional[str] = None,
    seed: int = 42,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Установи datasets: uv add datasets")

    run = None
    if wandb_project:
        try:
            import wandb
            run = wandb.init(
                project=wandb_project,
                name=f"longmemeval-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "benchmark": "LongMemEval",
                    "split": split,
                    "n_samples": n_samples,
                    "model": os.getenv("LLM_MODEL", "unknown"),
                    "memory_k": agent.memory_k,
                    "embed_model": agent.memory.EMBED_MODEL,
                    "seed": seed,
                },
            )
        except Exception as exc:
            logger.warning(f"W&B не запустился, продолжаю без него: {exc}")

    logger.info(f"Загружаю LongMemEval split={split}")
    dataset = load_dataset("xiaowu0162/longmemeval", split=split, trust_remote_code=True)

    import random
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:n_samples]
    subset = dataset.select(indices)

    correct = 0
    total = 0
    by_type: dict[str, list[float]] = {}
    rows = []

    for i, item in enumerate(subset):
        eval_uid = f"__eval_{i}_{int(time.time())}"
        sessions = item.get("sessions", [])
        question = item.get("question", "")
        answer = item.get("answer", "")
        qtype = item.get("question_type", "unknown")

        for session in sessions:
            if session:
                agent.ingest_conversation(eval_uid, session)
        agent.flush_background(timeout=60.0)

        try:
            response = agent.chat(eval_uid, "wellness_friend", question, stream=False)
        except Exception as exc:
            logger.error(f"bench item {i} ошибка: {exc}")
            response = ""

        score = _substring_score(answer, response)
        correct += int(score)
        total += 1
        by_type.setdefault(qtype, []).append(score)

        rows.append({
            "idx": i,
            "type": qtype,
            "question": question[:80],
            "expected": answer[:80],
            "got": response[:120],
            "score": score,
        })

        logger.info(f"bench [{i+1}/{len(subset)}] type={qtype} score={score:.0f}")

        agent.forget(eval_uid, "all")

    accuracy = correct / total if total else 0.0
    by_type_agg = {k: sum(v) / len(v) for k, v in by_type.items()}

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_type": by_type_agg,
    }

    if run is not None:
        try:
            import wandb
            run.log({"accuracy": accuracy, **{f"acc_{k}": v for k, v in by_type_agg.items()}})
            table = wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])
            run.log({"results": table})
            run.finish()
        except Exception as exc:
            logger.warning(f"W&B логирование не удалось: {exc}")

    logger.info(f"LongMemEval готово: accuracy={accuracy:.3f} ({correct}/{total})")
    return metrics
