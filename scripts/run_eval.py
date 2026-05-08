from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from mindly.agent import MindlyAgent
from mindly.eval.benchmark import run_longmemeval
from mindly.memory import MemoryLayer

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--split", default="oracle")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    memory = MemoryLayer(persist_dir="./data/eval_memory")
    agent = MindlyAgent(
        memory=memory,
        llm_client=client,
        model=os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct:free"),
    )

    wandb_project = os.getenv("WANDB_PROJECT", "mindly-eval") if args.wandb else None

    print(f"Запускаю LongMemEval: n={args.n}, split={args.split}, wandb={args.wandb}")
    metrics = run_longmemeval(
        agent=agent,
        n_samples=args.n,
        split=args.split,
        wandb_project=wandb_project,
    )

    print("\n" + "="*50)
    print("Результаты LongMemEval")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.3f}  ({metrics['correct']}/{metrics['total']})")
    print("\nПо типам вопросов:")
    for qtype, acc in sorted(metrics["by_type"].items()):
        print(f"  {qtype:<30} {acc:.3f}")

    import shutil
    shutil.rmtree("./data/eval_memory", ignore_errors=True)


if __name__ == "__main__":
    main()
