from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.base_agent import Agent
from agent.llm import LLMClient
from agent.scratchpad import Scratchpad
from agent.tools import BacktraderSandbox, Tavily, TwStock

# Pipeline stage labels
STAGES = [
    ("1/4 規劃", "Researcher 正在分析任務並制定研究計畫..."),
    ("2/4 研究", "正在蒐集市場數據與新聞..."),
    ("3/4 分析", "LLM 正在分析研究結果與市場情緒..."),
    ("4/4 審查", "Risk Manager 正在進行風險審查..."),
]


AVAILABLE_MODELS = {
    "claude-sonnet-4-5": "Anthropic Claude Sonnet 4.5",
    "claude-opus-4": "Anthropic Claude Opus 4",
    "gpt-4o": "OpenAI GPT-4o",
    "gpt-4o-mini": "OpenAI GPT-4o Mini",
    "gemini-2.5-pro": "Google Gemini 2.5 Pro",
    "deepseek-r1": "DeepSeek R1",
    "llama-4-maverick": "Meta Llama 4 Maverick",
}

DEFAULT_MODEL = "gpt-4o-mini"


class QuantPythonAgent:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.model = DEFAULT_MODEL
        self.scratchpad = Scratchpad()
        self.tavily = Tavily()
        self.llm = LLMClient(model=self.model)
        self.researcher = Agent(role="Researcher", tools=[self.tavily, TwStock])
        self.coder = Agent(role="Backtester", tools=[BacktraderSandbox])
        self.critic = Agent(role="Risk_Manager")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> None:
        table = Table(title="Available Models", show_header=True, header_style="bold cyan")
        table.add_column("ID")
        table.add_column("Description")
        table.add_column("Active", justify="center")
        for model_id, desc in AVAILABLE_MODELS.items():
            marker = "[green]●[/green]" if model_id == self.model else ""
            table.add_row(model_id, desc, marker)
        self.console.print(table)

    def set_model(self, model_id: str) -> None:
        if model_id not in AVAILABLE_MODELS:
            self.console.print(f"[red]Unknown model:[/red] {model_id}")
            self.console.print(f"[dim]Use /models to see available options[/dim]")
            return
        self.model = model_id
        self.llm.model = model_id
        self.console.print(f"[green]✓[/green] Model switched to [bold]{model_id}[/bold]")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_mission(self, query: str):
        self.scratchpad = Scratchpad()  # reset per mission
        c = self.console

        c.rule(f"[bold cyan]Mission: {query}")

        # Stage 1 – Planning
        plan = self._stage(0, self._plan, query)

        # Stage 2 – Research (Tavily)
        research = self._stage(1, self._research, plan)

        # Stage 3 – LLM Analysis
        analysis = self._stage(2, self._analyze, query, research)

        # Stage 4 – LLM Risk Review
        verdict = self._stage(3, self._review, query, analysis)

        # Final report
        self._print_report(query, research, analysis, verdict)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _plan(self, query: str) -> list[str]:
        """Return a list of research steps."""
        return [
            f"{query} 最新新聞",
            f"{query} 財務報告",
        ]

    def _research(self, steps: list[str]) -> dict:
        """Execute each research step via Tavily."""
        results = {}
        for step in steps:
            hits = self.tavily.search_news(step, max_results=5)
            for hit in hits:
                self.scratchpad.add({"type": "news", **hit})
            results[step] = hits
        return results

    def _analyze(self, query: str, research: dict) -> str:
        """Feed research into LLM for financial analysis + sentiment."""
        context = self._format_research_context(research)
        return self.llm.chat(
            system=(
                "你是一位專業的金融分析師。根據提供的新聞與研究資料，"
                "針對使用者的問題進行深度分析。請包含：\n"
                "1. 關鍵財務數據摘要\n"
                "2. 市場情緒判斷（看多/看空/中性），並說明理由\n"
                "3. 短期展望\n"
                "請用繁體中文回答，保持專業但易懂。"
            ),
            user=f"問題：{query}\n\n研究資料：\n{context}",
        )

    def _review(self, query: str, analysis: str) -> str:
        """Risk manager reviews the analysis."""
        return self.llm.chat(
            system=(
                "你是一位風險管理專家。審閱以下分析報告，補充：\n"
                "1. 潛在風險因素\n"
                "2. 風險等級（低/中/高）\n"
                "3. 建議的風險控制措施\n"
                "請用繁體中文回答，簡潔扼要。"
            ),
            user=f"原始問題：{query}\n\n分析報告：\n{analysis}",
        )

    @staticmethod
    def _format_research_context(research: dict) -> str:
        """Flatten research hits into a text block for the LLM."""
        parts = []
        for step, hits in research.items():
            parts.append(f"## {step}")
            for hit in hits:
                title = hit.get("title", "")
                content = hit.get("content", "")
                date = hit.get("published_date") or ""
                parts.append(f"- [{date}] {title}\n  {content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stage(self, idx: int, fn, *args):
        label, description = STAGES[idx]
        with self.console.status(f"[bold magenta][{label}][/bold magenta] {description}"):
            result = fn(*args)
        self.console.print(f"  [green]✓[/green] [bold]{label}[/bold] 完成")
        return result

    def _print_report(self, query, research, analysis, verdict):
        c = self.console

        # Research sources
        for step, hits in research.items():
            table = Table(
                title=step,
                show_header=True,
                header_style="bold cyan",
                show_lines=True,
                width=100,
            )
            table.add_column("#", width=3, justify="right")
            table.add_column("Title", width=35)
            table.add_column("Summary", width=50)
            table.add_column("Date", width=10)
            for i, hit in enumerate(hits, 1):
                table.add_row(
                    str(i),
                    hit.get("title", "")[:35],
                    hit.get("content", "")[:80],
                    (hit.get("published_date") or "")[:10],
                )
            c.print(table)
            c.print()

        # LLM Analysis
        c.print(Panel(Markdown(analysis), title="Financial Analysis", border_style="cyan"))

        # Risk Review
        c.print(Panel(Markdown(verdict), title="Risk Review", border_style="green"))

        c.rule("[bold green]Mission Complete")
