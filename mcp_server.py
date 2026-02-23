"""
quant-python-ai MCP Server

以 FastMCP 包裝 FinLab 量化工具，供 Claude Desktop 與其他 MCP Client 使用。

啟動方式：
    uv run mcp_server.py

Claude Desktop 設定（~/Library/Application Support/Claude/claude_desktop_config.json）：
    {
      "mcpServers": {
        "quant-python-ai": {
          "command": "uv",
          "args": ["--directory", "/Users/apple/quant-python-ai-main", "run", "mcp_server.py"]
        }
      }
    }
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agent.finlab_tools import (
    get_backtest_metrics,
    get_finlab_data,
    get_next_positions,
    list_strategies,
    optimize_conditions,
    run_backtest,
    run_liquidity_analysis,
)

load_dotenv()

mcp = FastMCP(
    name="quant-python-ai",
    instructions=(
        "你是一個台股量化交易助理，所有資料來自 FinMind API，與 FinLab 無關。\n"
        "重要限制：本系統只支援以下 4 種固定策略模板，不支援自訂策略：\n"
        "  1. revenue_growth  — 月營收連續成長\n"
        "  2. price_breakout  — 均線突破\n"
        "  3. high_dividend   — 高殖利率\n"
        "  4. momentum        — 動能策略\n"
        "使用 list_strategies_tool 可查看完整說明。\n"
        "若使用者要求 4 種之外的策略，請明確說明目前不支援，並建議最接近的現有策略。"
    ),
)


@mcp.tool()
def list_strategies_tool() -> str:
    """列出所有可用的量化策略模板，含策略 ID 與說明。

    Returns:
        策略清單 Markdown 格式
    """
    return list_strategies()


@mcp.tool()
def run_backtest_tool(
    strategy_id: str,
    resample: str = "M",
    stop_loss: float = 0.1,
    take_profit: float = 0.2,
    position_limit: float = 0.1,
) -> str:
    """執行指定策略的量化回測，傳回年化報酬、夏普率、最大回撤等績效摘要。

    重要：只支援以下 4 種固定策略 ID，不接受其他任何值：
    - revenue_growth：月營收連續成長策略
    - price_breakout：均線突破策略
    - high_dividend：高殖利率策略
    - momentum：動能策略

    Args:
        strategy_id: 策略模板 ID，必須是上述 4 種之一
        resample: 調倉頻率 D（每日）/ W（每週）/ M（每月）/ Q（每季），預設 M
        stop_loss: 停損比例，例如 0.1 代表 10%，預設 0.1
        take_profit: 停利比例，例如 0.2 代表 20%，預設 0.2
        position_limit: 單一持股上限比例，例如 0.1 代表 10%，預設 0.1

    Returns:
        回測績效摘要 Markdown 格式
    """
    return run_backtest(strategy_id, resample, stop_loss, take_profit, position_limit)


@mcp.tool()
def get_backtest_metrics_tool() -> str:
    """取得最近一次回測的完整績效指標。

    Returns:
        績效指標 Markdown 格式，若尚未回測則提示先執行 run_backtest_tool
    """
    return get_backtest_metrics()


@mcp.tool()
def get_next_positions_tool() -> str:
    """取得最近一次回測建議的下期換股清單，可作為實盤下單參考。

    Returns:
        換股清單 Markdown 格式（含股票代碼與建議持倉比例）
    """
    return get_next_positions()


@mcp.tool()
def run_liquidity_analysis_tool() -> str:
    """對最近一次回測結果執行流動性風險分析（漲跌停、成交量、處置股）。

    Returns:
        流動性分析摘要 Markdown 格式
    """
    return run_liquidity_analysis()


@mcp.tool()
def get_market_data_tool(
    dataset: str,
    start_date: str = "",
    end_date: str = "",
) -> str:
    """從 FinMind API 取得台股市場資料集的最新資料預覽。資料來源為 FinMind，非 FinLab。

    Args:
        dataset: 資料集名稱，格式為 'type:stock_id'，例如：
                 'price:2330'（台積電股價）、'revenue:2330'（月營收）、
                 'dividend:2330'（殖利率）
        start_date: 開始日期 YYYY-MM-DD（選填）
        end_date: 結束日期 YYYY-MM-DD（選填）

    Returns:
        資料集最新 10 行預覽（Markdown 表格格式），資料來自 FinMind API
    """
    return get_finlab_data(
        dataset,
        start_date=start_date or None,
        end_date=end_date or None,
    )


@mcp.tool()
def optimize_conditions_tool(
    strategy_id: str,
    resample: str = "M",
) -> str:
    """對策略模板執行參數條件優化，比較不同條件組合的績效，找出最佳參數。

    Args:
        strategy_id: 策略模板 ID（支援 revenue_growth / price_breakout）
        resample: 調倉頻率 D / W / M / Q，預設 M

    Returns:
        優化結果排行（依夏普率排序，前 5 名參數組合）Markdown 格式
    """
    return optimize_conditions(strategy_id, resample)


if __name__ == "__main__":
    mcp.run(transport="stdio")
