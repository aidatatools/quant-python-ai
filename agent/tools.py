import os

from tavily import TavilyClient


class Tavily:
    """Web/news search via Tavily API."""

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set in environment")
        self._client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        *,
        topic: str = "general",
        max_results: int = 5,
        time_range: str | None = None,
    ) -> list[dict]:
        """Run a search and return a list of result dicts."""
        kwargs: dict = {
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "include_answer": "basic",
        }
        if time_range:
            kwargs["time_range"] = time_range

        response = self._client.search(**kwargs)
        return self._parse(response)

    def search_news(
        self,
        query: str,
        *,
        max_results: int = 5,
        time_range: str = "month",
    ) -> list[dict]:
        """Shortcut for news-topic search."""
        return self.search(
            query,
            topic="news",
            max_results=max_results,
            time_range=time_range,
        )

    def search_finance(
        self,
        query: str,
        *,
        max_results: int = 5,
        time_range: str | None = None,
    ) -> list[dict]:
        """Shortcut for finance-topic search."""
        return self.search(
            query,
            topic="finance",
            max_results=max_results,
            time_range=time_range,
        )

    @staticmethod
    def _parse(response: dict) -> list[dict]:
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "published_date": r.get("published_date"),
            })
        return results


class TwStock:
    """Wrapper around twstock for fetching Taiwan stock data."""
    pass


class BacktraderSandbox:
    """Sandboxed backtrader execution environment."""
    pass
