"""
FinMind 向量化回測引擎

以 FinMind API 為唯一資料源，使用純 pandas/numpy 執行策略回測。
無需 FinLab API Token。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ── 預設股票宇宙 ───────────────────────────────────────────────────────────────
# 涵蓋電子、金融、傳產，流動性佳的上市股票
_DEFAULT_UNIVERSE = [
    # 電子/半導體
    "2330", "2317", "2454", "2382", "2308",
    "2303", "2357", "3711", "2379", "2395",
    "2377", "2301", "2344", "3034", "2049",
    # 金融
    "2891", "2882", "2886", "2884", "2885",
    "2892", "2880", "2881", "5880", "2883",
    # 傳產/消費
    "1301", "1303", "6505", "2912", "2207",
    "1216", "2105", "2002", "1402", "2603",
    # 科技服務
    "3008", "2409", "2408", "3481", "2376",
    "2352", "2327", "2353", "3045", "4938",
    # 其他大型股
    "2412", "2474", "2615", "2609", "2610",
]


def get_default_universe() -> list[str]:
    """取得預設股票宇宙清單（約 50 支流動性佳的台股）。"""
    env_universe = os.getenv("BACKTEST_UNIVERSE", "")
    if env_universe:
        return [s.strip() for s in env_universe.split(",") if s.strip()]
    return _DEFAULT_UNIVERSE.copy()


def _get_finmind_loader():
    """建立並登入 FinMind DataLoader。"""
    from FinMind.data import DataLoader
    api = DataLoader()
    token = os.getenv("FINMIND_API_KEY", "")
    if token:
        api.login_by_token(api_token=token)
    return api


def fetch_prices(
    universe: list[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """從 FinMind 抓取多股股價，回傳收盤價矩陣 (dates × stocks)。

    Args:
        universe: 股票代碼清單
        start_date: 開始日期 YYYY-MM-DD
        end_date: 結束日期 YYYY-MM-DD，不填則到今日

    Returns:
        pd.DataFrame，index=日期，columns=股票代碼，值=收盤價
    """
    import time
    api = _get_finmind_loader()
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    frames = []
    fail_count = 0

    for i, stock_id in enumerate(universe):
        try:
            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["close"]].rename(columns={"close": stock_id})
            frames.append(df)
        except Exception:
            fail_count += 1
            continue
        # FinMind rate limit 保護：每次呼叫後短暫等待
        if i < len(universe) - 1:
            time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    price_matrix = pd.concat(frames, axis=1).sort_index()
    if end_date:
        price_matrix = price_matrix[price_matrix.index <= end_date]
    return price_matrix


def fetch_revenues(
    universe: list[str],
    start_date: str,
) -> pd.DataFrame:
    """從 FinMind 抓取多股月營收，回傳月營收矩陣 (months × stocks)。"""
    import time
    api = _get_finmind_loader()
    frames = []

    for i, stock_id in enumerate(universe):
        try:
            df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(
                df["revenue_year"].astype(str) + "-" +
                df["revenue_month"].astype(str).str.zfill(2) + "-01"
            )
            df = df.set_index("date")[["revenue"]].rename(columns={"revenue": stock_id})
            frames.append(df)
        except Exception:
            continue
        if i < len(universe) - 1:
            time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index()


def fetch_dividend_yields(
    universe: list[str],
    start_date: str,
) -> pd.DataFrame:
    """從 FinMind 抓取多股殖利率，回傳殖利率矩陣 (dates × stocks)。"""
    import time
    api = _get_finmind_loader()
    frames = []

    for i, stock_id in enumerate(universe):
        try:
            df = api.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start_date)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["dividend_yield"]].rename(
                columns={"dividend_yield": stock_id}
            )
            frames.append(df)
        except Exception:
            continue
        if i < len(universe) - 1:
            time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index()


def build_signal(
    strategy_id: str,
    universe: list[str],
    start_date: str,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """依策略 ID 建立選股信號矩陣 (dates × stocks)，值為 True/False。

    Args:
        strategy_id: revenue_growth / price_breakout / high_dividend / momentum
        universe: 股票代碼清單
        start_date: 回測起始日期
        params: 策略參數覆蓋

    Returns:
        pd.DataFrame，True 代表當期持有
    """
    params = params or {}

    if strategy_id == "revenue_growth":
        months = params.get("months", 3)
        rev = fetch_revenues(universe, start_date)
        if rev.empty:
            return pd.DataFrame()
        # 月營收連續 N 個月成長
        signal = rev > rev.shift(1)
        for i in range(2, months + 1):
            signal = signal & (rev.shift(i - 1) > rev.shift(i))
        return signal.fillna(False)

    elif strategy_id == "price_breakout":
        ma_days = params.get("ma_days", 60)
        # 需要更早的資料才能算均線
        early_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ma_days * 2)
        ).strftime("%Y-%m-%d")
        prices = fetch_prices(universe, early_start)
        if prices.empty:
            return pd.DataFrame()
        ma = prices.rolling(window=ma_days, min_periods=ma_days // 2).mean()
        # 今日收盤 > 均線 且 昨日收盤 <= 昨日均線（突破訊號）
        breakout = (prices > ma) & (prices.shift(1) <= ma.shift(1))
        # 持有：已突破後繼續持倉（直到跌破）
        signal = prices > ma
        return signal[signal.index >= start_date].fillna(False)

    elif strategy_id == "high_dividend":
        top_pct = params.get("top_pct", 0.2)
        # 需要更早資料取均值
        early_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)
        ).strftime("%Y-%m-%d")
        div_yield = fetch_dividend_yields(universe, early_start)
        if div_yield.empty:
            return pd.DataFrame()
        # 月底重新採樣，每月選殖利率排名前 top_pct 的股票
        monthly = div_yield.resample("ME").last()
        n_pick = max(1, int(monthly.shape[1] * top_pct))
        signal = monthly.apply(
            lambda row: row >= row.nlargest(n_pick).min(), axis=1
        ).reindex(div_yield.index, method="ffill")
        return signal[signal.index >= start_date].fillna(False)

    elif strategy_id == "momentum":
        lookback_months = params.get("lookback_months", 3)
        top_pct = params.get("top_pct", 0.2)
        lookback_days = lookback_months * 21
        early_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=lookback_days + 10)
        ).strftime("%Y-%m-%d")
        prices = fetch_prices(universe, early_start)
        if prices.empty:
            return pd.DataFrame()
        # 近期動能：lookback 期報酬率排名
        momentum = prices / prices.shift(lookback_days) - 1
        monthly = momentum.resample("ME").last()
        n_pick = max(1, int(monthly.shape[1] * top_pct))
        signal = monthly.apply(
            lambda row: row >= row.nlargest(n_pick).min(), axis=1
        ).reindex(prices.index, method="ffill")
        return signal[signal.index >= start_date].fillna(False)

    else:
        return pd.DataFrame()


def run_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    stop_loss: float = 0.1,
    take_profit: float = 0.2,
    fee_ratio: float = 0.001425,
    tax_ratio: float = 0.003,
) -> dict:
    """向量化等權重回測。

    Args:
        signal: 持股信號矩陣 (dates × stocks)，True=持有
        prices: 收盤價矩陣 (dates × stocks)
        stop_loss: 停損比例
        take_profit: 停利比例
        fee_ratio: 手續費率（單邊）
        tax_ratio: 交易稅率

    Returns:
        dict 含 portfolio_returns、trade_count 等中間資料
    """
    # 對齊股票欄位
    common_stocks = signal.columns.intersection(prices.columns)
    if common_stocks.empty:
        raise ValueError("信號矩陣與股價矩陣沒有共同的股票代碼。")

    px = prices[common_stocks].ffill().copy()

    # 將信號（可能是月頻）reindex 到股價的日頻索引，前向填充
    sig = (
        signal[common_stocks]
        .reindex(px.index, method="ffill")
        .fillna(False)
        .astype(float)
        .copy()
    )

    # 等權重配置
    n_held = sig.sum(axis=1).replace(0, np.nan)
    weights = sig.div(n_held, axis=0).fillna(0.0)

    # 每日對數報酬（停牌或無成交日設為 0）
    pct = px.pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    log_ret = np.log1p(pct)

    # 持倉報酬（使用前一期權重，次日開盤）
    port_log_ret = (weights.shift(1).fillna(0.0) * log_ret).sum(axis=1)
    port_ret = np.expm1(port_log_ret)  # expm1 比 exp()-1 數值更穩定

    # 換手率 → 計算交易成本
    weight_change = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    transaction_cost = weight_change * (fee_ratio + tax_ratio / 2)
    port_ret_net = port_ret - transaction_cost

    # 累積報酬曲線
    cumulative = (1 + port_ret_net).cumprod()

    return {
        "daily_returns": port_ret_net,
        "cumulative": cumulative,
        "weights": weights,
        "n_stocks_held": n_held.fillna(0),
    }


def calc_metrics(backtest_result: dict) -> dict:
    """計算績效指標。

    Returns:
        dict 含 annualized_return, sharpe_ratio, max_drawdown, win_rate, total_return
    """
    daily_ret = backtest_result["daily_returns"].dropna()
    cumulative = backtest_result["cumulative"].dropna()

    if daily_ret.empty or len(daily_ret) < 2:
        return {}

    n_days = len(daily_ret)
    trading_days = 252

    total_return = float(cumulative.iloc[-1] - 1)
    annualized_return = float((1 + total_return) ** (trading_days / n_days) - 1)

    std = daily_ret.std()
    sharpe = float(
        (daily_ret.mean() * trading_days) /
        (std * np.sqrt(trading_days))
    ) if std > 0 else 0.0

    # 最大回撤
    roll_max = cumulative.cummax()
    drawdown = (cumulative - roll_max) / roll_max
    max_drawdown = float(drawdown.min())

    # 勝率（日報酬 > 0 的比例）
    win_rate = float((daily_ret > 0).mean())

    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
        "trading_days": n_days,
    }
