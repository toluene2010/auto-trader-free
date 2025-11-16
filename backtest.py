# backtest.py
import numpy as np
import pandas as pd

def compute_returns(df: pd.DataFrame, price_col="Close"):
    df = df.copy()
    df["ret"] = df[price_col].pct_change().fillna(0)
    return df

def performance_metrics(returns: pd.Series, periods_per_year=252):
    """
    returns: pd.Series of periodic returns (e.g., daily returns)
    periods_per_year: 252 for daily, 252*6.5*? for intraday depending on interval
    """
    rets = returns.dropna()
    if len(rets) == 0:
        return {}
    avg = rets.mean() * periods_per_year
    vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = avg / vol if vol != 0 else np.nan
    # Sortino
    negative = rets[rets < 0]
    downside = negative.std() * np.sqrt(periods_per_year) if len(negative) > 0 else np.nan
    sortino = avg / downside if downside != 0 and not np.isnan(downside) else np.nan
    # cumulative
    cumulative = (1 + rets).cumprod()
    total_return = cumulative.iloc[-1] - 1
    # annualized return
    ann_return = (1 + total_return) ** (periods_per_year / len(rets)) - 1 if len(rets) > 0 else np.nan
    # max drawdown
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    return {
        "annualized_return": ann_return,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "total_return": total_return,
        "max_drawdown": max_dd
    }

def vectorized_backtest_signals(df: pd.DataFrame, signals: pd.Series):
    """
    df: price dataframe with 'Close'
    signals: series aligned with df (1 -> long, 0 -> flat)
    Strategy: full allocation to long when signal==1, cash otherwise.
    Returns returns series of strategy
    """
    df = df.copy().loc[signals.index]
    prices = df["Close"]
    # position pct = signals (0 or 1)
    pos = signals.shift(1).fillna(0)  # enter next bar
    returns = prices.pct_change().fillna(0)
    strat_returns = pos * returns
    return strat_returns
