import os
from datetime import datetime, timedelta

from FinMind.data import DataLoader
from tavily import TavilyClient


class Tavily:
    """Web/news search via Tavily API."""

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set in environment")
        self._client = TavilyClient(api_key=api_key)

    def search_news(
        self,
        query: str,
        *,
        max_results: int = 5,
        time_range: str = "month",
    ) -> list[dict]:
        """搜尋近期新聞。"""
        response = self._client.search(
            query=query,
            topic="news",
            max_results=max_results,
            time_range=time_range,
        )
        return self._parse(response)

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

    @classmethod
    def get_schemas(cls) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_news",
                    "description": "搜尋近期新聞、財報相關資訊或市場事件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜尋關鍵字，例如 '台積電 營收'"},
                            "max_results": {"type": "integer", "description": "回傳結果數量", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]


class FinMindTool:
    """台灣市場數據工具，使用 FinMind API。"""

    def __init__(self):
        self.api_key = os.getenv("FINMIND_API_KEY", "")
        self.api = DataLoader()
        if self.api_key:
            self.api.login_by_token(api_token=self.api_key)

    def _get_start_date(self, days: int = 30) -> str:
        return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    def get_stock_price(self, stock_id: str, days: int = 30) -> str:
        """取得個股日成交資訊（股價、成交量）。"""
        df = self.api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=self._get_start_date(days)
        )
        return df.tail(10).to_markdown() if not df.empty else "無資料"

    def get_institutional_investors(self, stock_id: str, days: int = 30) -> str:
        """取得三大法人買賣超資料。"""
        df = self.api.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=self._get_start_date(days)
        )
        return df.tail(10).to_markdown() if not df.empty else "無資料"

    def get_margin_trading(self, stock_id: str, days: int = 30) -> str:
        """取得融資融券資料。"""
        df = self.api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id,
            start_date=self._get_start_date(days)
        )
        return df.tail(10).to_markdown() if not df.empty else "無資料"

    def get_month_revenue(self, stock_id: str, days: int = 180) -> str:
        """取得月營收資料。"""
        df = self.api.taiwan_stock_month_revenue(
            stock_id=stock_id,
            start_date=self._get_start_date(days)
        )
        return df.tail(6).to_markdown() if not df.empty else "無資料"

    def get_per_pbr(self, stock_id: str, days: int = 30) -> str:
        """取得個股本益比 (PER)、股價淨值比 (PBR) 與殖利率。"""
        # SDK 方法對應 dataset: TaiwanStockPER
        df = self.api.taiwan_stock_per_pbr(
            stock_id=stock_id,
            start_date=self._get_start_date(days)
        )
        return df.tail(10).to_markdown() if not df.empty else "無資料"

    def get_dividend_policy(self, stock_id: str) -> str:
        """取得股利政策（配股、配息）。"""
        # 使用 dataset: TaiwanStockDividend
        df = self.api.taiwan_stock_dividend(
            stock_id=stock_id,
            start_date=self._get_start_date(365 * 3)  # 預設看三年
        )
        return df.to_markdown() if not df.empty else "無資料"

    def get_financial_statements(self, stock_id: str, statement_type: str = "IncomeStatement") -> str:
        """取得財務報表（IncomeStatement, BalanceSheet, CashFlows）。"""
        start_date = self._get_start_date(365 * 2) # 預設看兩年
        if statement_type == "IncomeStatement":
            df = self.api.taiwan_stock_financial_statements(stock_id=stock_id, start_date=start_date)
        elif statement_type == "BalanceSheet":
            df = self.api.taiwan_stock_balance_sheet(stock_id=stock_id, start_date=start_date)
        elif statement_type == "CashFlows":
            df = self.api.taiwan_stock_cash_flows(stock_id=stock_id, start_date=start_date)
        else:
            return "錯誤：未知的報表類型。請使用 IncomeStatement, BalanceSheet 或 CashFlows。"
            
        return df.tail(20).to_markdown() if not df.empty else "無資料"

    @classmethod
    def get_schemas(cls) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "取得台灣個股日成交資訊（股價、成交量、漲跌幅）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"},
                            "days": {"type": "integer", "description": "查詢天數", "default": 30}
                        },
                        "required": ["stock_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_institutional_investors",
                    "description": "取得台灣個股三大法人（外資、投信、自營商）買賣超資料。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"},
                            "days": {"type": "integer", "description": "查詢天數", "default": 30}
                        },
                        "required": ["stock_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_margin_trading",
                    "description": "取得台灣個股融資融券變動資料。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"},
                            "days": {"type": "integer", "description": "查詢天數", "default": 30}
                        },
                        "required": ["stock_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_month_revenue",
                    "description": "取得台灣個股月營收資料（年增率、月增率）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"},
                            "days": {"type": "integer", "description": "查詢天數 (建議 180 天以上看趨勢)", "default": 180}
                        },
                        "required": ["stock_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_per_pbr",
                    "description": "取得台灣個股本益比 (PER)、股價淨值比 (PBR) 與殖利率資料。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"},
                            "days": {"type": "integer", "description": "查詢天數", "default": 30}
                        },
                        "required": ["stock_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dividend_policy",
                    "description": "取得台灣個股歷史股利政策（配股、配息）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"}
                        },
                        "required": ["stock_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financial_statements",
                    "description": "取得台灣個股財務報表數據。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string", "description": "股票代碼 (例如: 2330)"},
                            "statement_type": {
                                "type": "string", 
                                "enum": ["IncomeStatement", "BalanceSheet", "CashFlows"],
                                "description": "報表類型：損益表 (IncomeStatement)、資產負債表 (BalanceSheet)、現金流量表 (CashFlows)"
                            }
                        },
                        "required": ["stock_id"]
                    }
                }
            }
        ]


class BacktraderSandbox:
    """Sandboxed backtrader execution environment (Placeholder)."""
