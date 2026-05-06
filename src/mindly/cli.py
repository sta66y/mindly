from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .agent import MindlyAgent
from .memory import MemoryLayer
from .personas import PERSONA_KEYS, PERSONAS

load_dotenv()

app = typer.Typer(help="Mindly — AI велнес-коуч с персистентной памятью", add_completion=False)
console = Console()

DATA_DIR = Path(os.getenv("MEMORY_DIR", "./data/memory")).parent
LOG_FILE = DATA_DIR / "log_file.log"


def _setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(str(LOG_FILE), rotation="10 MB", retention="30 days", level="DEBUG", enqueue=True)
    logger.add(sys.stderr, level="WARNING", colorize=True)


def _build_agent() -> MindlyAgent:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct:free")
    memory_dir = os.getenv("MEMORY_DIR", "./data/memory")

    client = OpenAI(api_key=api_key, base_url=base_url)
    memory = MemoryLayer(persist_dir=memory_dir)
    return MindlyAgent(memory=memory, llm_client=client, model=model)


@app.command()
def chat(
    user: str = typer.Option(..., "--user", "-u", prompt="Имя пользователя"),
    persona: str = typer.Option(
        "wellness_friend",
        "--persona", "-p",
        help=f"Персона: {', '.join(PERSONA_KEYS)}",
    ),
) -> None:
    """Запустить интерактивный чат."""
    _setup_logging()

    if persona not in PERSONAS:
        console.print(f"[red]Неизвестная персона. Доступные: {', '.join(PERSONA_KEYS)}[/red]")
        raise typer.Exit(1)

    agent = _build_agent()
    current_persona = persona

    console.print(Panel(
        f"[bold cyan]Mindly[/bold cyan]  ·  {PERSONAS[current_persona]['name']}\n"
        f"Пользователь: [green]{user}[/green]\n\n"
        f"[dim]{PERSONAS[current_persona]['description']}[/dim]\n\n"
        "[bold]Команды:[/bold]\n"
        "  [yellow]/forget <тема>[/yellow]  — удалить воспоминания по теме\n"
        "  [yellow]/forget all[/yellow]     — удалить все воспоминания\n"
        "  [yellow]/persona <имя>[/yellow]  — сменить персону\n"
        "  [yellow]/memories[/yellow]       — показать сохранённые факты\n"
        "  [yellow]/quit[/yellow]           — выйти",
        title="Добро пожаловать",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = Prompt.ask(f"\n[green]{user}[/green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]До свидания![/dim]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped == "/quit":
            console.print("[dim]До свидания![/dim]")
            break

        if stripped.startswith("/forget "):
            query = stripped[8:].strip()
            agent.forget(user, query)
            if query == "all":
                console.print("[yellow]Все воспоминания удалены.[/yellow]")
            else:
                console.print(f"[yellow]Воспоминания по теме '{query}' удалены.[/yellow]")
            continue

        if stripped == "/memories":
            facts = agent.memory.list_all(user)
            if not facts:
                console.print("[dim]Пока ничего не сохранено.[/dim]")
            else:
                tbl = Table("Категория", "Факт", "Не поднимать", show_header=True)
                for f in facts:
                    tbl.add_row(
                        f["category"],
                        f["fact"],
                        "🔕" if f["do_not_raise"] else "",
                    )
                console.print(tbl)
            continue

        if stripped.startswith("/persona "):
            new_persona = stripped[9:].strip()
            if new_persona not in PERSONAS:
                console.print(f"[red]Неизвестная персона. Доступные: {', '.join(PERSONA_KEYS)}[/red]")
            else:
                current_persona = new_persona
                console.print(f"[green]Переключено на {PERSONAS[new_persona]['name']}[/green]")
            continue

        console.print(f"\n[bold]{PERSONAS[current_persona]['name']}[/bold]: ", end="")
        try:
            for token in agent.chat(user, current_persona, stripped, stream=True):
                console.print(token, end="", highlight=False)
        except Exception as exc:
            console.print(f"\n[red]Ошибка: {exc}[/red]")
            logger.error(f"chat error: {exc}")
        console.print()


@app.command()
def forget(
    user: str = typer.Option(..., "--user", "-u", prompt="Имя пользователя"),
    query: str = typer.Argument("all", help="Тема для удаления, или 'all' чтобы стереть всё"),
) -> None:
    """Удалить воспоминания пользователя из командной строки."""
    _setup_logging()
    agent = _build_agent()
    agent.forget(user, query)
    console.print(f"[yellow]Готово. Воспоминания пользователя '{user}' обновлены.[/yellow]")


@app.command()
def list_personas() -> None:
    """Показать доступные персоны."""
    for key, info in PERSONAS.items():
        console.print(f"[bold cyan]{key}[/bold cyan] — {info['name']}: {info['description']}")
