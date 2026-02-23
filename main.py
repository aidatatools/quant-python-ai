import sys

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from agent.quant_python_agent import QuantPythonAgent
from agent.quant_strategy_agent import QuantStrategyAgent

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
    console.print(Panel(BANNER, style="bold cyan", subtitle="v0.2.0"))
    console.print("[dim]輸入投資研究任務，或輸入 /help 查看指令、/quit 離開[/dim]\n")
    console.print("[dim]量化回測：/quant backtest <策略描述>　換股清單：/quant positions[/dim]\n")

    agent = QuantPythonAgent(console)
    quant_agent = QuantStrategyAgent(console, llm=agent.llm)
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
                    "[bold]/model[/bold] [dim]<provider:model>[/dim] - 切換模型\n"
                    "[bold]/quit[/bold]   - 離開程式\n"
                    "[bold]/help[/bold]   - 顯示此說明\n\n"
                    "── 研究任務（FinMind + Tavily）──\n"
                    '  "整理台積電近期重大事件與財務重點"\n'
                    '  "比較台積電和聯發科的近期新聞情緒差異"\n\n'
                    "── 量化策略（FinLab）──\n"
                    "[bold]/quant backtest[/bold] [dim]<描述>[/dim] - 執行量化回測\n"
                    "[bold]/quant positions[/bold]             - 查看最近回測換股清單\n"
                    "[bold]/quant help[/bold]                  - 量化工具說明",
                    title="Help",
                )
            )
            continue

        # ── /quant 指令：量化策略 Agent ────────────────────────────────────────
        if query.startswith("/quant"):
            parts = query.split(maxsplit=1)
            sub = parts[1].strip() if len(parts) > 1 else "help"

            if sub == "help":
                console.print(
                    Panel(
                        "[bold]/quant backtest[/bold] [dim]<策略描述>[/dim]\n"
                        "  執行量化回測，例如：/quant backtest 月營收連續成長的電子股，月度調倉\n\n"
                        "[bold]/quant positions[/bold]\n"
                        "  顯示最近一次回測建議的換股清單\n\n"
                        "可用策略模板：revenue_growth / price_breakout / high_dividend / momentum",
                        title="Quant 量化指令說明",
                        border_style="cyan",
                    )
                )
            elif sub == "positions":
                quant_agent.show_positions()
            elif sub.startswith("backtest"):
                task = sub[len("backtest"):].strip() or "請列出可用策略後，執行月營收連續成長策略回測"
                quant_agent.run(task)
            else:
                console.print(f"[yellow]未知 /quant 子指令：{sub}。輸入 /quant help 查看說明。[/yellow]")
            console.print()
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
