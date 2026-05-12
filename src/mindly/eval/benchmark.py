from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from loguru import logger

from ..agent import MindlyAgent


def _load_longmemeval(split: str = "oracle") -> list[dict]:
    """Download LongMemEval and return as a plain Python list.

    The HF repo stores files without extension and with nested structures that
    cause type-inference failures in Dataset.from_list — so we skip that and
    return the raw list directly.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("Установи huggingface_hub: uv add huggingface_hub")

    filename = f"longmemeval_{split}"
    logger.info(f"Скачиваю {filename} через hf_hub_download")
    local_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval",
        filename=filename,
        repo_type="dataset",
    )
    with open(local_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = next(iter(data.values())) if len(data) == 1 else list(data.values())[0]

    return data


def _substring_score(expected, predicted) -> float:
    return 1.0 if str(expected).strip().lower() in str(predicted).strip().lower() else 0.0


def run_longmemeval(
    agent: MindlyAgent,
    n_samples: int = 100,
    split: str = "oracle",
    wandb_project: Optional[str] = None,
    seed: int = 42,
) -> dict:
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
    dataset = _load_longmemeval(split)

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    subset = [dataset[i] for i in indices[:n_samples]]

    correct = 0
    total = 0
    by_type: dict[str, list[float]] = {}
    rows = []

    for i, item in enumerate(subset):
        eval_uid = f"__eval_{i}_{int(time.time())}"

        # The dataset field is 'haystack_sessions', each session is a list of turns.
        # Turns may include a 'has_answer' key — strip it before passing to the agent.
        sessions = item.get("haystack_sessions", [])
        question = item.get("question", "")
        answer = item.get("answer", "")
        qtype = item.get("question_type", "unknown")

        for session in sessions:
            if session:
                clean_turns = [
                    {"role": t["role"], "content": t["content"]}
                    for t in session
                    if isinstance(t, dict) and "role" in t and "content" in t
                ]
                if clean_turns:
                    agent.ingest_conversation(eval_uid, clean_turns)
        agent.flush_background(timeout=60.0)

        daily_limit_hit = False
        try:
            response = agent.chat(eval_uid, "wellness_friend", question, stream=False)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg and "per-day" in msg:
                logger.warning("Суточный лимит OpenRouter исчерпан — останавливаем бенчмарк досрочно")
                agent.forget(eval_uid, "all")
                daily_limit_hit = True
            else:
                logger.error(f"bench item {i} ошибка: {exc}")
            response = ""

        answer_str = str(answer)
        score = _substring_score(answer_str, response)
        correct += int(score)
        total += 1
        by_type.setdefault(qtype, []).append(score)

        rows.append({
            "idx": i,
            "type": qtype,
            "question": str(question)[:80],
            "expected": answer_str[:80],
            "got": response[:120],
            "score": score,
        })

        logger.info(f"bench [{i+1}/{len(subset)}] type={qtype} score={score:.0f}")
        agent.forget(eval_uid, "all")

        if daily_limit_hit:
            break

        logger.info(f"bench [{i+1}/{len(subset)}] type={qtype} score={score:.0f}")

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
