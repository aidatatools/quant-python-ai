class Scratchpad:
    """Agent memory scratchpad for accumulating research data."""

    def __init__(self):
        self._entries: list = []

    def add(self, data) -> None:
        self._entries.append(data)

    def get_news(self) -> list:
        return [e for e in self._entries if isinstance(e, dict) and e.get("type") == "news"]

    def summary(self) -> str:
        return "\n".join(str(e) for e in self._entries)
