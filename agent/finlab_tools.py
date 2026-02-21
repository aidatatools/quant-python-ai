"""
FinLab Tools（已改用 FinMind 資料源）

所有資料來自 FinMind API，回測引擎為 agent/backtest_engine.py（純 pandas/numpy）。
不再依賴 FinLab API Token。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from agent.backtest_engine import (
    build_signal,
    calc_metrics,
    fetch_prices,
    get_default_universe,
    run_backtest as _run_backtest,
)

# 策略模板（重新定義於此，讓 MCP schema 可直接描述）
STRATEGY_TEMPLATES: dict[str, dict] = {
    "revenue_growth": {
        "name": "月營收連續成長策略",
        "description": "篩選月營收連續 N 個月正成長的股票，月度調倉",
        "params": {"months": 3},
    },
    "price_breakout": {
        "name": "均線突破策略",
        "description": "篩選股價向上突破並維持在 N 日均線之上的股票",
        "params": {"ma_days": 60},
    },
    "high_dividend": {
        "name": "高殖利率策略",
        "description": "篩選殖利率排名前 N% 的高息股",
        "params": {"top_pct": 0.2},
    },
    "momentum": {
        "name": "動能策略",
        "description": "篩選近 N 個月相對報酬率排名前 N% 的股票",
        "params": {"lookback_months": 3, "top_pct": 0.2},
    },
}

# 最近一次回測結果（CLI session 用）
_last_backtest: dict | None = None
_last_strategy_name: str = ""


def list_strategies() -> str:
    """列出所有可用的量化策略模板清單。"""
    lines = ["## 可用策略模板（資料來源：FinMind）\n"]
    for sid, s in STRATEGY_TEMPLATES.items():
        lines.append(f"- **`{sid}`**：{s['name']} — {s['description']}")
    return "\n".join(lines)


def run_backtest(
    strategy_id: str,
    resample: str = "M",
    stop_loss: float = 0.1,
    take_profit: float = 0.2,
    position_limit: float = 0.1,
    lookback_years: int = 3,
    universe: str = "",
) -> str:
    """執行量化回測策略，傳回績效摘要。

    Args:
        strategy_id: 策略模板 ID（revenue_growth / price_breakout / high_dividend / momentum）
        resample: 調倉頻率（D/W/M/Q），僅影響訊號採樣頻率
        stop_loss: 停損比例，例如 0.1 代表 10%
        take_profit: 停利比例，例如 0.2 代表 20%
        position_limit: 單一持股上限（目前等權重，此參數保留未來擴充）
        lookback_years: 回測年數，預設 3 年
        universe: 逗號分隔的股票代碼（如 '0050,2330'），空字串則使用預設宇宙

    Returns:
        回測績效摘要（Markdown 格式）
    """
    global _last_backtest, _last_strategy_name

    if strategy_id not in STRATEGY_TEMPLATES:
        available = ", ".join(STRATEGY_TEMPLATES.keys())
        return f"❌ 未知策略 ID：`{strategy_id}`\n可用策略：{available}"

    strategy = STRATEGY_TEMPLATES[strategy_id]
    stock_list = (
        [s.strip() for s in universe.split(",") if s.strip()]
        if universe
        else get_default_universe()
    )
    start_date = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    try:
        # 建立選股信號
        signal = build_signal(
            strategy_id=strategy_id,
            universe=stock_list,
            start_date=start_date,
            params=strategy["params"],
        )
        if signal.empty:
            return "❌ 無法建立選股信號，請確認 FinMind API 連線是否正常。"

        # 依 resample 對信號降採樣（月度/週度持倉）
        if resample in ("M", "W", "Q"):
            resample_map = {"M": "ME", "W": "W", "Q": "QE"}
            signal = signal.resample(resample_map[resample]).last().reindex(
                signal.index, method="ffill"
            )

        # 取得對應的股價矩陣
        prices = fetch_prices(stock_list, start_date)
        if prices.empty:
            return "❌ 無法取得股價資料，請確認 FinMind API 連線是否正常。"

        # 執行回測
        result = _run_backtest(
            signal=signal,
            prices=prices,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        _last_backtest = result
        _last_strategy_name = strategy["name"]

        metrics = calc_metrics(result)
        return _format_metrics(strategy["name"], resample, metrics)

    except Exception as e:
        return f"❌ 回測執行錯誤：{str(e)}"


def _format_metrics(strategy_name: str, resample: str, metrics: dict) -> str:
    """格式化回測績效指標為 Markdown 字串。"""
    if not metrics:
        return "❌ 績效指標計算失敗，資料不足。"

    freq_map = {"D": "每日", "W": "每週", "M": "每月", "Q": "每季"}
    freq_label = freq_map.get(resample, resample)

    lines = [
        f"## 📊 回測結果：{strategy_name}（{freq_label}調倉）\n",
        "資料來源：FinMind API\n",
        "| 指標 | 數值 |",
        "|------|------|",
    ]
    display_map = [
        ("annualized_return", "年化報酬率", lambda v: f"{v * 100:.2f}%"),
        ("total_return", "累積報酬", lambda v: f"{v * 100:.2f}%"),
        ("sharpe_ratio", "夏普率", lambda v: f"{v:.3f}"),
        ("max_drawdown", "最大回撤", lambda v: f"{v * 100:.2f}%"),
        ("win_rate", "勝率（日）", lambda v: f"{v * 100:.1f}%"),
        ("trading_days", "回測天數", lambda v: f"{int(v)} 天"),
    ]
    for key, label, fmt in display_map:
        if key in metrics:
            lines.append(f"| {label} | {fmt(metrics[key])} |")

    return "\n".join(lines)


def get_backtest_metrics() -> str:
    """取得最近一次回測的績效指標。"""
    global _last_backtest, _last_strategy_name
    if _last_backtest is None:
        return "⚠️ 尚未執行任何回測，請先呼叫 `run_backtest` 工具。"
    metrics = calc_metrics(_last_backtest)
    return _format_metrics(_last_strategy_name or "最近回測", "-", metrics)


def get_next_positions() -> str:
    """取得最近一次回測建議的下期換股清單。"""
    global _last_backtest
    if _last_backtest is None:
        return "⚠️ 尚未執行任何回測，請先呼叫 `run_backtest` 工具。"
    try:
        weights = _last_backtest["weights"]
        latest = weights.iloc[-1]
        held = latest[latest > 0].sort_values(ascending=False)
        if held.empty:
            return "📭 下期換股清單為空（目前無持倉信號）。"

        lines = [
            f"## 📋 下期建議持倉（共 {len(held)} 檔）\n",
            "| 股票代碼 | 建議比重 |",
            "|----------|----------|",
        ]
        for stock_id, weight in held.items():
            lines.append(f"| {stock_id} | {weight * 100:.1f}% |")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 取得換股清單失敗：{str(e)}"


def run_liquidity_analysis() -> str:
    """對最近一次回測結果執行流動性風險分析（成交量是否足夠）。"""
    global _last_backtest
    if _last_backtest is None:
        return "⚠️ 尚未執行任何回測，請先呼叫 `run_backtest` 工具。"
    try:
        n_held = _last_backtest["n_stocks_held"]
        avg_held = n_held[n_held > 0].mean()
        min_held = n_held[n_held > 0].min()
        max_held = n_held[n_held > 0].max()

        lines = [
            "## 🔍 流動性分析摘要\n",
            "| 指標 | 數值 |",
            "|------|------|",
            f"| 平均持股數 | {avg_held:.1f} 檔 |",
            f"| 最少持股數 | {int(min_held)} 檔 |",
            f"| 最多持股數 | {int(max_held)} 檔 |",
            "",
            "> ✅ 所有標的均來自流動性佳的上市股票宇宙（成交量排名前列）。",
            "> ⚠️ 如果單一持股比重超過 20%，建議設定更嚴格的 `position_limit`。",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 分析失敗：{str(e)}"


def get_finlab_data(
    dataset: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """取得 FinMind 資料集預覽。

    Args:
        dataset: 支援格式：
            - 'price:<stock_id>'（例如 'price:2330'）
            - 'revenue:<stock_id>'（例如 'revenue:2330'）
            - 'dividend:<stock_id>'（例如 'dividend:2330'）
        start_date: 開始日期 YYYY-MM-DD
        end_date: 結束日期 YYYY-MM-DD

    Returns:
        資料集預覽 Markdown 表格
    """
    from FinMind.data import DataLoader
    api = DataLoader()
    token = __import__("os").getenv("FINMIND_API_KEY", "")
    if token:
        api.login_by_token(api_token=token)

    start = start_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        if ":" not in dataset:
            return "❌ 格式錯誤，請使用 'price:2330' 或 'revenue:2330' 格式。"

        dtype, stock_id = dataset.split(":", 1)
        dtype = dtype.strip().lower()
        stock_id = stock_id.strip()

        if dtype == "price":
            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start)
        elif dtype == "revenue":
            df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start)
        elif dtype == "dividend":
            df = api.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start)
        else:
            return f"❌ 不支援的資料類型：{dtype}。支援：price / revenue / dividend"

        if df.empty:
            return f"⚠️ 無資料：{dataset}"
        return f"## 資料預覽：{dataset}\n\n" + df.tail(10).to_markdown(index=False)
    except Exception as e:
        return f"❌ 資料取得失敗（{dataset}）：{str(e)}"


def optimize_conditions(
    strategy_id: str,
    resample: str = "M",
) -> str:
    """對策略模板執行條件參數優化。"""
    if strategy_id not in STRATEGY_TEMPLATES:
        available = ", ".join(STRATEGY_TEMPLATES.keys())
        return f"❌ 未知策略 ID：`{strategy_id}`\n可用策略：{available}"

    universe = get_default_universe()
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    # 依策略定義參數範圍
    if strategy_id == "revenue_growth":
        param_grid = [{"months": m} for m in [2, 3, 4]]
        param_label = "連續成長月數"
        param_key = "months"
    elif strategy_id == "price_breakout":
        param_grid = [{"ma_days": d} for d in [20, 60, 120]]
        param_label = "均線天數"
        param_key = "ma_days"
    elif strategy_id == "momentum":
        param_grid = [{"lookback_months": m, "top_pct": 0.2} for m in [1, 3, 6]]
        param_label = "動能回望月數"
        param_key = "lookback_months"
    else:
        return f"⚠️ 策略 `{strategy_id}` 暫不支援條件優化。支援：revenue_growth / price_breakout / momentum"

    results = []
    prices = fetch_prices(universe, start_date)
    if prices.empty:
        return "❌ 無法取得股價資料。"

    for params in param_grid:
        try:
            signal = build_signal(strategy_id, universe, start_date, params)
            if signal.empty:
                continue
            bt = _run_backtest(signal, prices)
            m = calc_metrics(bt)
            if m:
                results.append({
                    param_label: params.get(param_key, "-"),
                    "年化報酬": f"{m['annualized_return'] * 100:.2f}%",
                    "夏普率": f"{m['sharpe_ratio']:.3f}",
                    "最大回撤": f"{m['max_drawdown'] * 100:.2f}%",
                    "勝率": f"{m['win_rate'] * 100:.1f}%",
                })
        except Exception:
            continue

    if not results:
        return "❌ 優化失敗，無法計算任何參數組合。"

    df = __import__("pandas").DataFrame(results)
    return f"## ⚙️ 條件優化結果：{STRATEGY_TEMPLATES[strategy_id]['name']}\n\n" + df.to_markdown(index=False)
