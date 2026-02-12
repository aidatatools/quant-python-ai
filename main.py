import sys

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from agent.quant_python_agent import QuantPythonAgent

theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "stage": "bold magenta",
    }
)
console = Console(theme=theme)

BANNER = r"""
  ___                    _     ____        _   _                      _    ___
 / _ \ _   _  __ _ _ __ | |_  |  _ \ _   _| |_| |__   ___  _ __      / \  |_ _|
| | | | | | |/ _` | '_ \| __| | |_) | | | | __| '_ \ / _ \| '_ \    / _ \  | |
| |_| | |_| | (_| | | | | |_  |  __/| |_| | |_| | | | (_) | | | |  / ___ \ | |
 \__\_\\__,_|\__,_|_| |_|\__| |_|    \__, |\__|_| |_|\___/|_| |_| /_/   \_\___|
                                      |___/
"""


def main():
    load_dotenv()
    console.print(Panel(BANNER, style="bold cyan", subtitle="v0.1.0"))
    console.print("[dim]輸入投資研究任務，或輸入 /help 查看指令、/quit 離開[/dim]\n")

    agent = QuantPythonAgent(console)
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            query = session.prompt(">> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            sys.exit(0)

        query = query.strip()
        if not query:
            continue

        if query == "/quit":
            console.print("[dim]Bye![/dim]")
            break

        if query == "/help":
            console.print(
                Panel(
                    "[bold]/models[/bold]  - 列出可用模型\n"
                    "[bold]/model[/bold] [dim]<id>[/dim] - 切換模型\n"
                    "[bold]/quit[/bold]   - 離開程式\n"
                    "[bold]/help[/bold]   - 顯示此說明\n\n"
                    "直接輸入任務即可開始研究，例如:\n"
                    '  "給我台積電一月的財務報告 並判斷市場情緒"\n'
                    '  "比較 台積電 和 聯發科 的財務指標 誰更有可能具備更多的漲勢"',
                    title="Help",
                )
            )
            continue

        if query == "/models":
            agent.list_models()
            continue

        if query.startswith("/model"):
            parts = query.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[dim]Usage: /model <id>  (see /models for options)[/dim]")
            else:
                agent.set_model(parts[1])
            continue

        agent.run_mission(query)
        console.print()


if __name__ == "__main__":
    main()
