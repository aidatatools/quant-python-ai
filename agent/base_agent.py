class Agent:
    """Base agent that can plan, execute tools, and generate code."""

    def __init__(self, role: str, tools: list | None = None):
        self.role = role
        self.tools = tools or []

    def plan(self, query: str):
        """Create an execution plan from a user query."""
        raise NotImplementedError

    def execute_tool(self, step):
        """Execute a single research step using the assigned tools."""
        raise NotImplementedError

    def analyze_sentiment(self, news: list) -> float:
        """Return a sentiment score from news data."""
        raise NotImplementedError

    def generate_code(self, target: str, logic: str) -> str:
        """Generate backtest code from a trading logic description."""
        raise NotImplementedError

    def execute_in_sandbox(self, code: str):
        """Run generated code in a sandboxed environment."""
        raise NotImplementedError

    def review(self, research: str, backtest) -> dict:
        """Review research and backtest results (critic role)."""
        raise NotImplementedError
