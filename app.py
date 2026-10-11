"""Stock Rater: enter a ticker, get bull / base / bear cases and an overall rating.

Data: free Yahoo Finance data via yfinance. Every metric is scored from -1 (bearish) to +1 (bullish)
by linear interpolation between a "bad" and a "good" threshold, then rolled up by category.
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timezone

import research as R
import qual as Q
import sec as SEC
from catalog import CATALOG

HORIZONS = {  # category weights per time horizon (missing = 0)
    "Short-term": {"Valuation": .10, "Profitability": .10, "Growth": .10, "Financial health": .10,
                   "Technicals": .35, "Analyst": .20, "Risk": .05},
    "Long-term": {"Valuation": .20, "Profitability": .20, "Growth": .20, "Financial health": .15,
                  "Track record": .15, "Analyst": .05, "Risk": .05},
}

# (category, label, yfinance info key, "good" threshold, "bad" threshold)
# If good < bad, lower is better.
INFO_METRICS = [
    ("Valuation", "Trailing P/E", "trailingPE", 15, 35),
    ("Valuation", "Forward P/E", "forwardPE", 14, 32),
    ("Valuation", "PEG ratio", "pegRatio", 1, 2.5),
    ("Valuation", "Price / Book", "priceToBook", 2, 10),
    ("Valuation", "Price / Sales", "priceToSalesTrailing12Months", 2, 10),
    ("Valuation", "EV / EBITDA", "enterpriseToEbitda", 9, 25),
    ("Valuation", "EV / Revenue", "enterpriseToRevenue", 2, 10),
    ("Profitability", "Net margin", "profitMargins", .15, .03),
    ("Profitability", "Operating margin", "operatingMargins", .20, .05),
    ("Profitability", "Gross margin", "grossMargins", .50, .20),
    ("Profitability", "EBITDA margin", "ebitdaMargins", .25, .08),
    ("Profitability", "Return on equity", "returnOnEquity", .18, .04),
    ("Profitability", "Return on assets", "returnOnAssets", .08, .01),
    ("Growth", "Revenue growth (YoY)", "revenueGrowth", .12, -.02),
    ("Growth", "Earnings growth (YoY)", "earningsGrowth", .12, -.05),
    ("Growth", "Quarterly earnings growth", "earningsQuarterlyGrowth", .12, -.05),
    ("Financial health", "Current ratio", "currentRatio", 2, .9),
    ("Financial health", "Quick ratio", "quickRatio", 1.2, .6),
    ("Financial health", "Debt / Equity (%)", "debtToEquity", 40, 180),
    ("Analyst", "Analyst rating (1=Strong Buy, 5=Sell)", "recommendationMean", 2.0, 3.5),
    ("Risk", "Beta", "beta", .9, 1.8),
    ("Risk", "Short interest (% of float)", "shortPercentOfFloat", .02, .12),
    ("Risk", "Institutional ownership", "heldPercentInstitutions", .70, .20),
]
NEG_IS_BAD = {"trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda"}  # loss-making = bearish


CYCLICAL = {"Energy", "Basic Materials", "Industrials", "Consumer Cyclical"}
_FIN_SKIP = {"Debt / Equity (%)", "Current ratio", "Quick ratio", "Cash / Debt", "Net debt / EBITDA", "EV / EBITDA",
             "EV / Revenue", "Price / Sales", "Gross margin", "EBITDA margin", "Operating margin",
             "Free cash flow yield", "FCF / Net income", "Interest coverage", "Total debt change (YoY)",
             "Share of years with positive FCF", "Gross margin change (YoY)", "Operating margin change (YoY)",
             "Operating margin stability (std dev)"}
_DEF_GROWTH = {k: (.05, -.01) for k in ("Revenue growth (YoY)", "Revenue growth (latest fiscal yr)", "Revenue 3-yr CAGR")}
# Sector rules of thumb: "over" = (good, bad) thresholds replacing the generic ones, "skip" = metrics that don't
# mean much for the sector, "exit" = plausible long-run P/E range used in 5-year scenarios.
PROFILES = {
    "Technology": {"over": {"Trailing P/E": (25, 55), "Forward P/E": (22, 40), "Price / Sales": (5, 16), "Price / Book": (6, 20),
                            "EV / EBITDA": (18, 38), "EV / Revenue": (5, 14), "PEG ratio": (1.5, 3), "Gross margin": (.65, .40),
                            "Operating margin": (.25, .08), "Debt / Equity (%)": (40, 120)}, "skip": set(), "exit": (18, 32)},
    "Financial Services": {"over": {"Price / Book": (1.0, 2.5), "Trailing P/E": (10, 20), "Forward P/E": (10, 18),
                                    "Return on equity": (.13, .06), "Return on assets": (.012, .004), "Net margin": (.25, .10)},
                           "skip": _FIN_SKIP, "exit": (9, 16)},
    "Healthcare": {"over": {"Trailing P/E": (22, 45), "Forward P/E": (18, 32), "Gross margin": (.65, .35), "Operating margin": (.20, .02),
                            "Net margin": (.15, 0), "Price / Sales": (3, 10), "EV / EBITDA": (12, 25)}, "skip": set(), "exit": (16, 28)},
    "Consumer Cyclical": {"over": {"Trailing P/E": (15, 32), "Forward P/E": (14, 28), "Net margin": (.08, .02), "Operating margin": (.10, .03),
                                   "Gross margin": (.35, .15), "EV / EBITDA": (9, 18), "Price / Sales": (1, 4),
                                   "Debt / Equity (%)": (70, 200)}, "skip": set(), "exit": (12, 22)},
    "Consumer Defensive": {"over": {"Trailing P/E": (18, 30), "Forward P/E": (17, 26), "Net margin": (.07, .02), "Operating margin": (.12, .04),
                                    "Gross margin": (.35, .15), "EV / EBITDA": (11, 18), "Price / Sales": (1, 3),
                                    "Debt / Equity (%)": (80, 200), **_DEF_GROWTH}, "skip": set(), "exit": (16, 24)},
    "Industrials": {"over": {"Trailing P/E": (17, 32), "Forward P/E": (16, 28), "Net margin": (.09, .03), "Operating margin": (.14, .05),
                             "Gross margin": (.32, .15), "EV / EBITDA": (11, 20), "Price / Sales": (1.5, 4),
                             "Debt / Equity (%)": (70, 170)}, "skip": set(), "exit": (14, 22)},
    "Energy": {"over": {"Trailing P/E": (9, 20), "Forward P/E": (9, 18), "EV / EBITDA": (4.5, 9), "Price / Sales": (1, 3), "Net margin": (.10, .02),
                        "Operating margin": (.15, .03), "Gross margin": (.40, .15), "Debt / Equity (%)": (35, 100),
                        "Price / Book": (1.2, 3)}, "skip": set(), "exit": (8, 14)},
    "Utilities": {"over": {"Trailing P/E": (16, 25), "Forward P/E": (15, 22), "Price / Book": (1.6, 3), "EV / EBITDA": (10, 15),
                           "Debt / Equity (%)": (110, 260), "Net margin": (.12, .05), "Operating margin": (.20, .10),
                           "Return on equity": (.10, .05), "Current ratio": (.9, .5), "Quick ratio": (.6, .3),
                           "Interest coverage": (3.5, 1.8), "Net debt / EBITDA": (4.5, 7), **_DEF_GROWTH},
                  "skip": {"Free cash flow yield", "FCF / Net income", "Share of years with positive FCF", "Gross margin",
                           "Gross margin change (YoY)"}, "exit": (14, 20)},
    "Real Estate": {"over": {"Price / Book": (1.3, 3), "EV / EBITDA": (14, 25), "Debt / Equity (%)": (90, 220), "Net margin": (.20, .05),
                             "Return on equity": (.08, .02), "Interest coverage": (3, 1.5)},
                    "skip": {"Trailing P/E", "Forward P/E", "PEG ratio", "Free cash flow yield", "FCF / Net income", "Gross margin",
                             "Gross margin change (YoY)", "Current ratio", "Quick ratio"}, "exit": (14, 22)},
    "Basic Materials": {"over": {"Trailing P/E": (11, 25), "Forward P/E": (11, 22), "EV / EBITDA": (6, 12), "Price / Sales": (1.2, 3.5),
                                 "Net margin": (.10, .03), "Operating margin": (.15, .05), "Gross margin": (.30, .12),
                                 "Debt / Equity (%)": (35, 100)}, "skip": set(), "exit": (10, 17)},
    "Communication Services": {"over": {"Trailing P/E": (18, 35), "Forward P/E": (16, 28), "EV / EBITDA": (9, 18), "Operating margin": (.20, .08),
                                        "Net margin": (.12, .04), "Debt / Equity (%)": (70, 200), "Price / Sales": (2.5, 8)},
                               "skip": set(), "exit": (14, 26)},
}
PROFILES["Financial (asset-light)"] = {"over": {"Trailing P/E": (22, 40), "Forward P/E": (20, 34), "Operating margin": (.30, .10),
                                             "Net margin": (.25, .08), "Price / Book": (8, 25)}, "skip": set(), "exit": (20, 30)}
DEFAULT = {"over": {}, "skip": set(), "exit": (15, 25)}


def profile_for(info, sector):
    if sector == "Financial Services" and not R.use_book_value(info, sector):
        return PROFILES["Financial (asset-light)"]
    return PROFILES.get(sector, DEFAULT)


def lin(v, good, bad):
    return float(np.clip(2 * (v - bad) / (good - bad) - 1, -1, 1))


def row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s):
                return s
    return None


def growth(s):
    if s is None or len(s) < 2 or s.iloc[1] <= 0:
        return None
    return s.iloc[0] / s.iloc[1] - 1


def build_metrics(info, hist, inc, cf, bs, prof=None):
    prof = prof or DEFAULT
    rows = []

    def add(cat, label, v, good=None, bad=None, neg_bad=False, score=None):
        if label in prof["skip"]:
            return
        if label in prof["over"] and score is None:
            good, bad = prof["over"][label]
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if np.isnan(v) or np.isinf(v):
            return
        if score is None:
            score = -1.0 if (neg_bad and v <= 0) else lin(v, good, bad)
        rows.append({"Category": cat, "Metric": label, "Value": v, "Score": score})

    # --- Fundamentals from the info table ---
    for cat, label, key, good, bad in INFO_METRICS:
        add(cat, label, info.get(key), good, bad, key in NEG_IS_BAD)

    mc, fcf_ttm = info.get("marketCap"), info.get("freeCashflow")
    if mc and fcf_ttm:
        add("Valuation", "Free cash flow yield", fcf_ttm / mc, .05, 0)
    cash, debt_i, ebitda = info.get("totalCash"), info.get("totalDebt"), info.get("ebitda")
    if cash is not None and debt_i:
        add("Financial health", "Cash / Debt", cash / debt_i, 1, .2)
    if debt_i is not None and cash is not None and ebitda and ebitda > 0:
        add("Financial health", "Net debt / EBITDA", (debt_i - cash) / ebitda, 1, 4)
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    for key, label, g, b in (("targetMeanPrice", "Upside to mean analyst target", .15, -.05),
                             ("targetMedianPrice", "Upside to median analyst target", .15, -.05)):
        if price and info.get(key):
            add("Analyst", label, info[key] / price - 1, g, b)

    fe, te = info.get("forwardEps"), info.get("trailingEps")
    if fe and te and te > 0 and fe > 0:
        add("Growth", "Forward EPS growth", fe / te - 1, .12, 0)

    # --- Statement-derived trends ---
    rev = row(inc, ["Total Revenue", "Operating Revenue"])
    ni = row(inc, ["Net Income"])
    oi = row(inc, ["Operating Income"])
    gp = row(inc, ["Gross Profit"])
    ebit = row(inc, ["EBIT"])
    intx = row(inc, ["Interest Expense"])
    fcf = row(cf, ["Free Cash Flow"])
    sh = row(bs, ["Ordinary Shares Number", "Share Issued"])
    debt = row(bs, ["Total Debt"])

    add("Growth", "Revenue growth (latest fiscal yr)", growth(rev), .12, 0)
    if rev is not None and len(rev) >= 4 and rev.iloc[3] > 0 and rev.iloc[0] > 0:
        add("Growth", "Revenue 3-yr CAGR", (rev.iloc[0] / rev.iloc[3]) ** (1 / 3) - 1, .12, 0)
    add("Growth", "Net income growth (latest fiscal yr)", growth(ni), .12, 0)
    add("Growth", "Operating income growth", growth(oi), .12, 0)
    add("Growth", "Free cash flow growth", growth(fcf), .12, 0)
    if rev is not None and len(rev) >= 2:
        if gp is not None and len(gp) >= 2:
            add("Profitability", "Gross margin change (YoY)", gp.iloc[0] / rev.iloc[0] - gp.iloc[1] / rev.iloc[1], .01, -.02)
        if oi is not None and len(oi) >= 2:
            add("Profitability", "Operating margin change (YoY)", oi.iloc[0] / rev.iloc[0] - oi.iloc[1] / rev.iloc[1], .01, -.02)
    if fcf is not None and ni is not None and ni.iloc[0] > 0:
        add("Profitability", "FCF / Net income", fcf.iloc[0] / ni.iloc[0], 1, .5)
    if sh is not None and len(sh) >= 2:
        add("Financial health", "Share count change (buybacks < 0)", sh.iloc[0] / sh.iloc[1] - 1, -.01, .03)
    if debt is not None and len(debt) >= 2 and debt.iloc[1] > 0:
        add("Financial health", "Total debt change (YoY)", debt.iloc[0] / debt.iloc[1] - 1, -.05, .15)
    if ebit is not None and intx is not None and abs(intx.iloc[0]) > 0:
        add("Financial health", "Interest coverage", ebit.iloc[0] / abs(intx.iloc[0]), 8, 2)

    # --- Long-term track record from statements ---
    if rev is not None and len(rev) >= 3:
        add("Track record", "Share of years with revenue growth", (rev.diff(-1).dropna() > 0).mean(), 1, .5)
    if ni is not None and len(ni) >= 3:
        add("Track record", "Share of years profitable", (ni > 0).mean(), 1, .5)
    if fcf is not None and len(fcf) >= 3:
        add("Track record", "Share of years with positive FCF", (fcf > 0).mean(), 1, .5)
    if oi is not None and rev is not None:
        m = (oi / rev).dropna()
        if len(m) >= 3:
            add("Track record", "Operating margin stability (std dev)", m.std(), .02, .08)

    # --- Technicals and risk from price history ---
    c = hist["Close"].dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
    if len(c) > 60:
        last = c.iloc[-1]
        r = c.pct_change().dropna()
        for n in (20, 50, 100, 200):
            if len(c) > n:
                add("Technicals", f"Price vs {n}-day average", last / c.rolling(n).mean().iloc[-1] - 1, .05, -.05)
        if len(c) > 200:
            add("Technicals", "50-day vs 200-day average",
                c.rolling(50).mean().iloc[-1] / c.rolling(200).mean().iloc[-1] - 1, .03, -.03)
        for n, lab, g, b in ((21, "1-month", .05, -.05), (63, "3-month", .10, -.10),
                             (126, "6-month", .15, -.10), (252, "12-month", .25, -.10)):
            if len(c) > n:
                add("Technicals", f"{lab} return", last / c.iloc[-n - 1] - 1, g, b)
        d = c.diff()
        up, dn = d.clip(lower=0).rolling(14).mean().iloc[-1], (-d.clip(upper=0)).rolling(14).mean().iloc[-1]
        if dn and dn > 0:
            rsi = 100 - 100 / (1 + up / dn)
            add("Technicals", "RSI (14)", rsi, score=1.0 if rsi < 30 else (-1.0 if rsi > 75 else 0.0))
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        add("Technicals", "MACD histogram / price", (macd - macd.ewm(span=9).mean()).iloc[-1] / last, .01, -.01)
        yr = c.tail(252)
        add("Technicals", "Distance from 52-week high", last / yr.max() - 1, -.05, -.30)
        vol = r.tail(252).std() * np.sqrt(252)
        add("Risk", "Annualized volatility", vol, .25, .60)
        add("Risk", "Max drawdown (1 yr)", (yr / yr.cummax() - 1).min(), -.15, -.45)
        if vol > 0:
            add("Risk", "1-yr return / volatility", r.tail(252).mean() * 252 / vol, 1, 0)
    if len(c) > 756:  # 3+ years of history
        for yrs in (3, 5, 10):
            n = 252 * yrs
            if len(c) > n:
                add("Track record", f"{yrs}-yr annualized price return", (c.iloc[-1] / c.iloc[-n - 1]) ** (1 / yrs) - 1, .15, .03)
        for yrs in (5, 10):
            n = 252 * yrs
            if len(c) > n:
                win = c.tail(n)
                add("Track record", f"Max drawdown ({yrs} yr)", (win / win.cummax() - 1).min(), -.30, -.60)
        yearly = c.groupby(c.index.year).last().pct_change().dropna()
        if len(yearly) >= 3:
            add("Track record", "Share of calendar years up", (yearly > 0).mean(), .8, .5)
    return pd.DataFrame(rows)


def composite(df, weights):
    cs = df.groupby("Category")["Score"].mean()
    w = pd.Series(weights).reindex(cs.index).fillna(0)
    if w.sum() == 0:
        w[:] = 1
    return cs, float((cs * w).sum() / w.sum())


def label(final):
    if final > .35: return "STRONG BUY"
    if final > .15: return "BUY"
    if final > -.15: return "HOLD"
    if final > -.35: return "SELL"
    return "STRONG SELL"


def analyze(df, price, vol, info):
    """Short-term (about 12 months)."""
    weights = HORIZONS["Short-term"]
    cat_scores, comp = composite(df, weights)

    tm, th, tl = info.get("targetMeanPrice"), info.get("targetHighPrice"), info.get("targetLowPrice")
    base = tm if tm else price * (1 + .10 * comp)
    bull = min(th, price * (1 + 1.5 * vol)) if th else price * (1 + vol)
    bear = max(tl, price * (1 - 1.5 * vol)) if tl else price * (1 - vol)
    bull, bear = max(bull, base), min(bear, base)

    p_bull, p_bear = .25 + .25 * comp, .25 - .25 * comp
    p_base = 1 - p_bull - p_bear
    ev = p_bull * bull + p_base * base + p_bear * bear
    exp_ret = ev / price - 1

    final = .7 * comp + .3 * float(np.clip(exp_ret / .20, -1, 1))
    return dict(cat_scores=cat_scores, comp=comp, final=final, rating=label(final), exp_ret=exp_ret, weights=weights,
                scen=[("🐂 Bull", bull, p_bull), ("⚖️ Base", base, p_base), ("🐻 Bear", bear, p_bear)])


def analyze_long(df, price, info, prof=None):
    prof = prof or DEFAULT
    """Long-term (3-5 years): business quality + 5-year growth/valuation scenarios (price only, no dividends)."""
    weights = HORIZONS["Long-term"]
    cat_scores, comp = composite(df, weights)

    def val(name):
        m = df.loc[df.Metric == name, "Value"]
        return float(m.iloc[0]) if len(m) else None
    gs = [val(n) for n in ("Revenue 3-yr CAGR", "Revenue growth (YoY)", "Forward EPS growth", "Earnings growth (YoY)")]
    gs = [float(np.clip(g, -.10, .30)) for g in gs if g is not None]
    g0 = float(np.clip(np.median(gs), -.05, .18)) if gs else .05
    pe = info.get("forwardPE") or info.get("trailingPE")
    pe = float(pe) if pe and pe > 0 and "Trailing P/E" not in prof["skip"] else None
    lo, hi = prof["exit"]

    def proj(g, lo, hi):
        mult = float(np.clip(pe, lo, hi)) / pe if pe else 1.0
        return price * (1 + g) ** 5 * mult
    base = proj(g0, lo, hi)
    bull = proj(min(g0 + .05, .30), lo + 3, hi + 7)
    bear = price * (1 + max(g0 - .08, -.10)) ** 5 * (.65 if pe else 1.0)
    bull, bear = max(bull, base), min(bear, base)

    p_bull, p_bear = .25 + .25 * comp, .25 - .25 * comp
    p_base = 1 - p_bull - p_bear
    ev = p_bull * bull + p_base * base + p_bear * bear
    ann = lambda p: (p / price) ** (1 / 5) - 1
    exp_ret = ann(ev)
    final = .7 * comp + .3 * float(np.clip((exp_ret - .08) / .08, -1, 1))  # vs. an ~8%/yr market benchmark
    note = (f"5-year price targets. Base case: ~{g0:.0%}/yr growth with the P/E settling toward {lo}-{hi}x (typical for this sector). "
            "Bull: +5 pts growth and a richer multiple. Bear: -8 pts growth and a 35% lower multiple. "
            "Rating compares expected annual return with an 8%/yr market benchmark.")
    return dict(cat_scores=cat_scores, comp=comp, final=final, rating=label(final), exp_ret=exp_ret, weights=weights,
                scen=[("🐂 Bull", bull, p_bull, ann(bull)), ("⚖️ Base", base, p_base, ann(base)),
                      ("🐻 Bear", bear, p_bear, ann(bear))], note=note)


def thesis(df, info, short, long_, sector):
    """Reads the metrics together like an analyst: archetype, interactions between signals, adjustments."""
    cs = df.groupby("Category")["Score"].mean()
    g = lambda k: float(cs.get(k, 0.0))
    V, P, G, H, T, A, K = (g(k) for k in ("Valuation", "Profitability", "Growth", "Financial health",
                                          "Technicals", "Analyst", "Track record"))

    def mv(name):
        r = df.loc[df.Metric == name, "Value"]
        return float(r.iloc[0]) if len(r) else 0.0
    if P < -.2 and H < -.2: arch = "Turnaround / financially stretched"
    elif P > .35 and G > .2 and K > .2: arch = "Quality compounder"
    elif G > .4: arch = "Growth story"
    elif V > .35 and P < .3: arch = "Value / out-of-favor"
    elif sector in {"Utilities", "Real Estate", "Consumer Defensive"}: arch = "Defensive / income"
    elif sector in CYCLICAL: arch = "Cyclical"
    else: arch = "Balanced, no standout"

    # (condition, observation, long-term adj, short-term adj, what would change my mind)
    rules = [
        (V < -.3 and G > .4 and P > .3, "The valuation is rich, but growth and profitability are strong enough that the premium is defensible.", .05, 0, "Growth slowing would undercut the case for paying up."),
        (V < -.3 and G < .1, "I'd be paying a high price for little growth, a combination that tends to disappoint.", -.10, -.05, "A lower price, or growth reaccelerating."),
        (V > .3 and (G < -.2 or P < -.1), "It looks cheap, but shrinking growth or weak margins suggest it may be cheap for a reason (value-trap risk).", -.10, 0, "Evidence that sales and margins have stopped falling."),
        (V > .3 and P > .2 and G > -.1 and H > 0, "A reasonable price for a healthy, profitable business is the mismatch I look for.", .10, .05, "Margins slipping would remove the margin of safety."),
        (T > .4 and (P < 0 or G < 0), "The stock is rising faster than the business; momentum without fundamental support tends to fade.", 0, -.08, "Earnings or margins catching up with the price."),
        (T < -.3 and P > .3 and G > .2, "A strong business whose stock is out of favor; weak price action alone is not a reason to sell.", .05, 0, "The price stabilizing above its 200-day average would improve timing."),
        (H < -.3, "The balance sheet is the main risk: leverage magnifies any slowdown, so the bear case deserves extra weight.", -.10, -.05, "Falling debt or rising interest coverage."),
        (A > .5 and (P + G) / 2 < -.1, "Analysts are more optimistic than the underlying numbers justify, so I trust the numbers.", -.05, -.05, "Actual earnings beating estimates."),
        (mv("Net margin") < 0 and G > .3 and mv("Cash / Debt") > 1, "Unprofitable but growing and cash-funded; the question is whether it reaches profitability before the cash runs out.", 0, 0, "A visible path to positive free cash flow."),
        (mv("Operating margin change (YoY)") > .01, "Operating margins are expanding, a sign of operating leverage.", .03, .02, "Margin gains reversing."),
        (mv("Operating margin change (YoY)") < -.02, "Operating margins are compressing, which usually precedes weaker earnings.", -.05, -.03, "Margins stabilizing."),
        (sector in CYCLICAL, "Earnings in this sector swing with the cycle, so a low P/E can signal peak earnings, not a bargain.", 0, 0, "Where we are in the sector's cycle."),
    ]
    notes, watch, dl, ds = [], [], 0.0, 0.0
    for cond, text, a, b, w in rules:
        if cond:
            notes.append(text + (f" (long {a:+.2f}, short {b:+.2f})" if a or b else ""))
            watch.append(w); dl += a; ds += b
    spread = float(cs.std()) if len(cs) > 1 else 0.0
    conf = "High" if len(df) >= 35 and spread < .45 else ("Low" if len(df) < 22 or spread > .7 else "Medium")
    why = f"Confidence {conf.lower()}: {len(df)} metrics scored; categories {'mostly agree' if spread < .45 else 'disagree notably'}."
    strong = [f"{k} ({v:+.2f})" for k, v in cs.sort_values(ascending=False).items() if v > .25][:3]
    weak = [f"{k} ({v:+.2f})" for k, v in cs.sort_values().items() if v < -.15][:3]
    return dict(arch=arch, notes=notes, watch=watch, conf=conf, why=why, strong=strong, weak=weak,
                adj_long=float(np.clip(dl, -.3, .3)), adj_short=float(np.clip(ds, -.3, .3)))


@st.cache_data(ttl=3600, show_spinner=False)
def load(sym):
    t = yf.Ticker(sym)

    def safe(f):
        try:
            return f()
        except Exception:
            return None
    hist = safe(lambda: t.history(period="10y", auto_adjust=True))
    est = {k: safe(f) for k, f in (("rev", lambda: t.revenue_estimate), ("eps", lambda: t.earnings_estimate),
                                   ("trend", lambda: t.eps_trend), ("hist", lambda: t.earnings_history))}
    spy = safe(lambda: yf.Ticker("SPY").history(period="5y", auto_adjust=True))
    rf = safe(lambda: float(yf.Ticker("^TNX").history(period="5d")["Close"].dropna().iloc[-1]) / 100)
    extra = {"est": est, "spy": spy, "rf": rf, "fetched": datetime.now(timezone.utc)}
    return (safe(lambda: t.info) or {}, hist, safe(lambda: t.income_stmt),
            safe(lambda: t.cashflow), safe(lambda: t.balance_sheet), extra)


@st.cache_data(ttl=86400, show_spinner=False)
def load_peers(sym, industry_key):
    try:
        syms = [s for s in yf.Industry(industry_key).top_companies.index if s != sym][:6]
    except Exception:
        return None
    rows = []
    for s in syms:
        try:
            i = yf.Ticker(s).info
            rows.append({"Ticker": s, "Fwd P/E": i.get("forwardPE"), "EV/EBITDA": i.get("enterpriseToEbitda"),
                         "P/S": i.get("priceToSalesTrailing12Months"), "Op margin": i.get("operatingMargins"),
                         "Rev growth": i.get("revenueGrowth")})
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else None


def fmt(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def render(res, df, price):
    st.write(f"Metric-model score: **{res['final']:+.2f}** (an input to the recommendation, not a separate rating)")
    st.subheader("Scenarios")
    rows = []
    for n, p, pr, *ann in res["scen"]:
        r = {"Case": n, "Target": f"${p:,.2f}", "vs. now": f"{p / price - 1:+.1%}"}
        if ann:
            r["Per year"] = f"{ann[0]:+.1%}"
        r["Probability"] = f"{pr:.0%}"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.write(f"Probability-weighted expected {'annual return' if len(res['scen'][0]) > 3 else 'move'}: "
             f"**{res['exp_ret']:+.1%}**")
    if res.get("note"):
        st.caption(res["note"])
    used = [k for k, v in res["weights"].items() if v > 0]
    st.subheader("Category scores")
    st.bar_chart(res["cat_scores"].reindex(used).dropna().round(2))
    d = df[df.Category.isin(used)]
    st.subheader("🐂 Bull case: what's working")
    for _, m in d[d.Score > .3].sort_values("Score", ascending=False).head(6).iterrows():
        st.write(f"🟢 {m.Metric}: {fmt(m.Value)}")
    st.subheader("🐻 Bear case: what's worrying")
    for _, m in d[d.Score < -.3].sort_values("Score").head(6).iterrows():
        st.write(f"🔴 {m.Metric}: {fmt(m.Value)}")


S2C = {"Technology": 3, "Communication Services": 1.2, "Healthcare": 1.5, "Consumer Cyclical": 2, "Consumer Defensive": 2,
       "Industrials": 1.5, "Energy": .8, "Basic Materials": .8, "Utilities": .4, "Real Estate": .3}
NOT_EVALUATED = ["AI opportunity / disruption scoring (needs filing and transcript text analysed by an LLM)",
                 "Competitive moat and management quality (qualitative, needs transcripts and filings)",
                 "SEC XBRL, earnings-call transcripts, segment and geographic revenue, backlog and retention KPIs",
                 "Options-implied volatility, credit ratings, insider-transaction analysis",
                 "Industry operating indicators (wafer starts, ARR, comps, NIM, etc.)",
                 "Portfolio analyzer, screener across a universe, database persistence, background jobs"]


def _g(df, r, c):
    try:
        return float(df.loc[r, c])
    except Exception:
        return None


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.1%}"


def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        import os
        return os.environ.get(name)


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec(sym, ua):
    """Official SEC XBRL statements for US filers. Returns {'err': reason} when unavailable; never raises."""
    tj, err = SEC.get_json(SEC.TICKERS, ua)
    if err:
        return {"err": f"SEC ticker list unreachable ({err})"}
    cik = SEC.cik_for(sym, tj)
    if cik is None:
        return {"err": "ticker not in the SEC filer list (ETF, non-US company, or foreign private issuer filing 20-F/40-F)"}
    facts, err = SEC.get_json(SEC.BASE.format(cik=cik), ua)
    if err:
        return {"err": f"SEC company facts unreachable ({err})"}
    inc, cf, bs, meta = SEC.normalize(facts)
    return {"inc": inc, "cf": cf, "bs": bs, "meta": meta, "err": None if inc is not None else "no annual 10-K revenue facts in the filing data"}


def _years(df):
    r = R.getrow(df, ["Total Revenue", "Operating Revenue"])
    return 0 if r is None else len(r)


def prepare(sym):
    """One place that decides which source feeds each input, and records it. Never invents a value."""
    info, hist, inc, cf, bs, extra = load(sym)
    info, extra = dict(info), dict(extra)
    src, conflicts = [], []
    ua = _secret("SEC_USER_AGENT") or "StockRater/1.0 (set SEC_USER_AGENT in app secrets: app name + your email)"
    sd = load_sec(sym, ua)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or (float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None)
    yahoo_years = _years(inc)
    sec_ok = sd.get("inc") is not None and _years(sd["inc"]) >= 3 and _years(sd["inc"]) >= yahoo_years
    if sec_ok:
        inc, cf, bs = sd["inc"], sd["cf"], sd["bs"]
        m = sd["meta"]
        src.append(("Financial statements", "SEC EDGAR XBRL (official)", f"{_years(inc)} fiscal years; latest FY ending {m['latest_fy_end']}, filed {m['filed']}"))
        if m.get("age_days", 0) > 450:
            conflicts.append(f"SEC statements are {m['age_days']} days old (latest 10-K FY {m['latest_fy_end']}); newer quarters are not included.")
        fb, notes = SEC.fallback_info(m, inc, cf, bs, price)
        yrev = info.get("totalRevenue")
        for k, v in fb.items():
            if info.get(k) in (None, 0):
                info[k] = v; src.append((k, "SEC EDGAR", notes.get(k, "latest fiscal year")))
        if yrev and fb.get("totalRevenue") and not .6 <= yrev / fb["totalRevenue"] <= 1.6:
            conflicts.append(f"Yahoo TTM revenue ({yrev:,.0f}) and SEC latest-FY revenue ({fb['totalRevenue']:,.0f}) differ by more than 60%: check for a recent acquisition, fiscal-year change or units mismatch.")
    else:
        src.append(("Financial statements", "Yahoo Finance via yfinance (unofficial)", f"{yahoo_years} fiscal years" + (f"; SEC unavailable: {sd.get('err')}" if sd.get("err") else "")))
    src.append(("Price and history", "Yahoo Finance via yfinance (delayed)", f"last close {hist.index[-1]:%Y-%m-%d}" if hist is not None and not hist.empty else "none returned"))
    src.append(("Analyst estimates", "Yahoo Finance via yfinance" if any(v is not None for v in extra["est"].values()) else "unavailable", "consensus; no free source is fully reliable"))
    if info.get("beta") is None and hist is not None and not hist.empty:
        q = R.quant(hist["Close"], extra["spy"])
        if "Beta vs SPY" in q:
            info["beta"] = q["Beta vs SPY"]; src.append(("Beta", "computed from price history vs SPY", "Yahoo returned none"))
    src.append(("Risk-free rate", "Yahoo ^TNX (live)" if extra["rf"] else "assumption 4.2% (live rate unavailable)", ""))
    extra["sources"], extra["conflicts"], extra["sec_status"] = src, conflicts, ("used" if sec_ok else (sd.get("err") or "not used (fewer fiscal years than Yahoo)"))
    return info, hist, inc, cf, bs, extra


def coverage(df, prof):
    have = set(df.Metric) if "Metric" in df.columns else set()
    excl = [m for _, m in CATALOG if m in prof["skip"] and m not in have]
    ev = [m for _, m in CATALOG if m in have]
    unav = [m for _, m in CATALOG if m not in have and m not in prof["skip"]]
    return dict(total=len(CATALOG), evaluated=ev, excluded=excl, unavailable=unav)


def show_missing(d, kind, extra, info, price, mult):
    """Replaces a blank page: says exactly what is missing and why, and shows any partial result that is still valid."""
    st.warning("The intrinsic valuation could not be completed for this ticker. Nothing was estimated or filled in.")
    if kind == "dcf":
        rows = [("Revenue (TTM or latest fiscal year)", d["rev0"]), ("Operating margin", d["m0"]), ("Shares outstanding (market cap / price)", d["shares"]), ("Target operating margin", d["mt"])]
    else:
        rows = [("Book value per share", info.get("bookValue")), ("Return on equity", info.get("returnOnEquity"))]
    st.dataframe(pd.DataFrame([{"Required input": n, "Status": "available" if v not in (None, 0) else "MISSING", "Value": "n/a" if v in (None, 0) else f"{v:,.4g}"} for n, v in rows]),
                 hide_index=True, width="stretch")
    for x in d.get("issues", []):
        st.write(f"• {x}")
    st.write(f"• SEC statements: {extra['sec_status']}")
    st.write("• Common causes: an ETF or fund, a foreign company (SEC covers US 10-K filers only), a very new listing, or Yahoo throttling its response. "
             "Retrying after a minute often helps; setting SEC_USER_AGENT in the app secrets enables the SEC source.")
    if mult:
        st.write("**Partial result that is still valid:** forward EPS × sector P/E range")
        st.dataframe(pd.DataFrame([{"Case": k, "Fair value": f"${v['fv']:,.2f}", "vs price": pct(v["fv"] / price - 1)} for k, v in mult.items()]), hide_index=True, width="stretch")


def research_ui(sym, info, hist, inc, cf, bs, df, extra, sector, price, short, long_, th):
    watch = th["watch"]
    est = extra["est"]
    kind = "pb" if R.use_book_value(info, sector) else "dcf"
    rev = R.getrow(inc, ["Total Revenue", "Operating Revenue"])
    oi = R.getrow(inc, ["Operating Income"])
    eg = _g(est.get("rev"), "+1y", "growth") if est.get("rev") is not None else None
    d = R.derive(info, price, inc, bs, sector, extra["rf"], eg, sector in CYCLICAL, S2C,
                 sum(profile_for(info, sector)["over"].get("Operating margin", (.12, .12))) / 2)
    rev0, m0, shares, nd, mi, hist_g, avg_m = (d[k] for k in ("rev0", "m0", "shares", "nd", "mi", "hist_g", "avg_m"))
    rf, rf_live, beta, ke0, wacc0, g1_def, mt_def = (d[k] for k in ("rf", "rf_live", "beta", "ke", "wacc", "g1", "mt"))
    dcf_ok = kind == "dcf" and d["dcf_ok"]
    pb_ok = kind == "pb" and info.get("bookValue") and info.get("returnOnEquity")

    with st.expander("⚙️ Assumptions & scoring weights (change anything; everything recomputes)"):
        if dcf_ok:
            g1 = st.slider("Revenue growth, early years (%, fades to terminal)", -10.0, 40.0, round(g1_def * 100, 1), .5, key=f"g1{sym}") / 100
            mt = st.slider("Target operating margin (%)", -10.0, 50.0, round(mt_def * 100, 1), .5, key=f"mt{sym}") / 100
            wacc = st.slider("WACC (%)", 5.0, 15.0, round(wacc0 * 100, 1), .1, key=f"wc{sym}") / 100
            tg = st.slider("Terminal growth (%)", 0.0, 4.0, round(float(np.clip(rf * .6, .02, .03)) * 100, 1), .1, key=f"tg{sym}") / 100
            s2c = st.slider("Sales per $ of invested capital", .2, 5.0, float(S2C.get(sector, 1.5)), .1, key=f"s2c{sym}")
            ke = ke0
        elif pb_ok:
            roe = st.slider("Sustainable ROE (%)", 0.0, 40.0, round(float(np.clip(info["returnOnEquity"], 0, .4)) * 100, 1), .5, key=f"roe{sym}") / 100
            ke = st.slider("Cost of equity (%)", 5.0, 15.0, round(ke0 * 100, 1), .1, key=f"ke{sym}") / 100
            tg = st.slider("Long-run growth (%)", 0.0, 4.0, 2.5, .1, key=f"gg{sym}") / 100
        else:
            st.info("Not enough reported data for an intrinsic valuation; the verdict uses metrics only.")
            ke = ke0
        w = {k: st.slider(k + " weight (long-term scorecard)", 0, 100, int(v * 100), key=f"w{k}") for k, v in R.DEFAULT_W.items()}
        w_short = st.slider("Weight on the short-term outlook in the overall verdict (%)", 0, 100, 40, key="wshort")
        w_qual = st.slider("Weight on qualitative analysis (%), scaled by how much of it has evidence", 0, 100, 50, key="wqual")

    cs = df.groupby("Category")["Score"].mean().to_dict()
    q = float(np.clip((cs.get("Profitability", 0) + cs.get("Growth", 0) + cs.get("Financial health", 0)) / 3, -1, 1))
    inp = dict(price=price, q=q, ke=ke, hist_g=hist_g, cyc=sector in CYCLICAL, dcf_w=(.2 if sector == "Utilities" else .4 if sector in CYCLICAL else .6),
               fwd_eps=info.get("forwardEps") if d["cur_ok"] else None,
               pe_rng=profile_for(info, sector)["exit"])
    if dcf_ok:
        inp["p"] = dict(rev0=rev0, g1=g1, m0=m0, mt=mt, wacc=wacc, tg=tg, tax=.21, s2c=s2c, net_debt=nd + mi, shares=shares)
    elif pb_ok:
        inp["pb"] = dict(bvps=float(info["bookValue"]), roe=roe, g=tg)
    V = R.valuate(inp)
    scen, impl, plaus, base_dcf, p, ret, val, disagree, mult, pw = (V[k] for k in (
        "scen", "impl", "plaus", "base_dcf", "p", "ret", "val", "disagree", "mult", "pw"))
    flags, fm = R.forensic(inc, cf, bs)
    mv = lambda n: (lambda r: float(r.iloc[0]) if len(r) else None)(df.loc[df.Metric == n, "Value"])
    ctx = dict(rev_g=info.get("revenueGrowth"), margin_chg=mv("Operating margin change (YoY)"), eps_g=info.get("earningsGrowth"),
               oi_g=mv("Operating income growth"), share_chg=fm.get("Share count change (YoY)"), cash_conv=fm.get("Cash conversion (CFO / net income)"),
               fwd_pe=info.get("forwardPE"), cyclical=sector in CYCLICAL, ps=info.get("priceToSalesTrailing12Months"),
               margin_peak=(m0 is not None and avg_m is not None and m0 > avg_m * 1.15) if m0 is not None else None,
               gross_m=info.get("grossMargins"), fcf_yield=mv("Free cash flow yield"), tv_share=base_dcf["tv_share"] if base_dcf else None,
               quality=None, exp5=ret[5] if ret else None, ke=ke, analyst=cs.get("Analyst"),
               rev_trend=(lambda t: (t["current"] / t["90daysAgo"] - 1) if t is not None else None)(
                   est["trend"].loc["+1y"] if est.get("trend") is not None and "+1y" in est["trend"].index else None))
    nflags = sum(1 for f in flags if f[0] == "high") + sum(1 for f in flags if f[0] == "med") // 2 + (2 if d["speculative"] else 0)
    qitems = Q.assess_data(inc, cf, bs, info, V, wacc=(wacc if dcf_ok else wacc0))
    ai = st.session_state.get(f"ai_{sym}")
    if ai:
        for k_, it_ in ai["items"].items():
            if qitems[k_]["score"] is None:
                qitems[k_] = it_
    qscore, qcov = Q.summarize(qitems)
    override = ai["challenge"] if ai and ai.get("challenge") and st.session_state.get(f"ai_apply_{sym}") else None
    rec = R.recommend(cs, V, ke, short["comp"], len(df), nflags, {k: v / 100 for k, v in w.items()}, w_short / 100,
                      th["adj_short"], th["adj_long"], qual=dict(score=qscore, coverage=qcov), w_qual=w_qual / 100, override=override)
    sc, total, contrib = rec["sc"], rec["card"], rec["contrib"]
    ctx["quality"] = sc["Business quality"]
    cons = R.contradictions(ctx)
    verdict = rec["overall"]["rating"]
    pw = sum(v["prob"] * v["fv"] for v in scen.values()) if scen else None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    warns = R.validate(info, hist, price, now)

    st.header(f"{info.get('shortName', sym)} ({sym})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Short-term (3-12 mo)", rec["short"]["rating"], f"{rec['short']['score']:+.2f}", delta_color="off")
    c2.metric("Long-term (3-5 yr)", rec["long"]["rating"], f"{rec['long']['score']:+.2f}", delta_color="off")
    c3.metric("Overall", rec["overall"]["rating"], f"{rec['overall']['score']:+.2f}", delta_color="off")
    c4.metric("Price", f"${price:,.2f}")
    st.write(rec["summary"])
    st.caption(f"Quantitative vs qualitative: qualitative view {qscore:+.2f} with {qcov:.0%} rubric coverage, so it carries {rec['qual_weight']['long']:.0%} of the long-term "
               f"and {rec['qual_weight']['short']:.0%} of the short-term score (quantitative only: long {rec['quant_only']['long']:+.2f}, short {rec['quant_only']['short']:+.2f}).")
    if override:
        st.warning(f"AI analyst override applied ({override['adjustment']:+.2f}): {override['claim']} Evidence: {override['evidence']}")
    for x in extra["conflicts"]:
        st.warning(x)
    for c in rec["constraints"]:
        st.warning(f"**{c['horizon']} constrained: {c['rule']}.** {c['detail'].capitalize()}. Rating moved {c['from']} → {c['to']}.")
    st.caption(f"Sector: {sector or 'n/a (generic thresholds)'} · Profile: {th['arch']} · Confidence: {rec['conf']:.0%}")
    if th["strong"]: st.write("**Strongest areas:** " + ", ".join(th["strong"]))
    if th["weak"]: st.write("**Weakest areas:** " + ", ".join(th["weak"]))
    for n in th["notes"]:
        st.write(f"• {n}")
    st.caption(th["why"] + " Thresholds are sector rules of thumb, not live peer medians.")

    tabs = st.tabs(["🏁 Verdict", "💰 Valuation", "🎲 Scenarios", "🔍 Quality & flags", "📊 Quant & peers", "⏱ Metric model", "📋 Data & limits", "🧪 Audit", "🧠 Qualitative"])
    with tabs[0]:
        st.subheader(f"Overall verdict: {verdict}")
        with st.expander("How these ratings are built (one framework for all three)"):
            st.markdown("Every rating uses the same thresholds: >+0.40 Strong Buy, >+0.15 Buy, >-0.15 Hold, >-0.40 Avoid, else Strong Avoid. "
                        "**Short-term (3-12 mo)** = 40% near-term metric model + 35% modeled 12-month excess return + 25% downside risk. "
                        "**Long-term (3-5 yr)** = the scorecard below (quality, growth, valuation, 5-year excess return, downside, financial strength). "
                        "Both pass the same constraints (required-return floor, evidence-quality gate). **Overall** = weighted average of the two "
                        f"constrained scores ({w_short}% short / {100 - w_short}% long), so it can never fall outside the two horizon ratings. "
                        "Text never decides a rating.")
            st.dataframe(pd.DataFrame([(n_, f"{a_:.0%}", f"{b_:+.2f}") for n_, a_, b_ in rec["short_parts"]] +
                                      [("Interaction adjustment (rules)", "", f"{rec['adj_short']:+.2f}")], columns=["Short-term input", "Weight", "Value"]), hide_index=True, width="stretch")
        for c in rec["constraints"]:
            st.warning(f"{c['horizon']}: {c['rule']}: {c['detail']} ({c['from']} → {c['to']})")
        stamp = hist.index[-1].strftime("%Y-%m-%d")
        st.write(f"Price **${price:,.2f}** (last close {stamp}, Yahoo Finance via yfinance, delayed; fetched {extra['fetched']:%Y-%m-%d %H:%M} UTC)")
        if scen:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bear", f"${scen['Bear']['fv']:,.0f}", pct(scen["Bear"]["fv"] / price - 1))
            c2.metric("Base", f"${scen['Base']['fv']:,.0f}", pct(scen["Base"]["fv"] / price - 1))
            c3.metric("Bull", f"${scen['Bull']['fv']:,.0f}", pct(scen["Bull"]["fv"] / price - 1))
            c4.metric("Prob.-weighted", f"${pw:,.0f}", pct(pw / price - 1))
            st.write(f"Modeled annualized return (price converging to fair value over ~3 yrs, value compounding at {ke:.1%}): "
                     f"**1y {pct(ret[1])} · 3y {pct(ret[3])} · 5y {pct(ret[5])}**. Margin of safety: **{pct(1 - price / pw)}**. "
                     f"Looks attractive below ~**${pw * .8:,.0f}** (20% margin of safety); unattractive above ~**${pw * 1.15:,.0f}**.")
        else:
            st.write("No intrinsic valuation available for this ticker, so the verdict rests on metrics only.")
        rows = [{"Score": k, "Value": "not evaluated" if v is None else (f"{v:.2f} (0-1)" if k == "Forecast confidence" else f"{v:+.2f}"),
                 "Weight": f"{w[k]}%" if k in w else "", "Contribution": f"{contrib[k]:+.3f}" if k in contrib else ""} for k, v in sc.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"This scorecard (weighted {total:+.2f}) is the long-term analysis; it feeds the long-term rating. Weights are adjustable above.")
        st.markdown("**What the market is pricing in:** " + (f"the price implies about **{impl:+.1%}** revenue growth for the early years (assessed **{plaus.lower()}** vs "
                    f"history of {pct(hist_g)}), given the margin, WACC and terminal assumptions above." if impl is not None else "not computable (no DCF solution or no DCF for this sector)."))
        st.markdown("**Strongest case against owning it:** " + (cons[0] if cons else (flags[0][1] + ": " + flags[0][2] if flags else (
            f"the bear case of ${scen['Bear']['fv']:,.0f} ({pct(scen['Bear']['fv'] / price - 1)})." if scen else "no specific red flag detected."))))
        st.markdown("**What would change my mind:** " + (" ".join(watch[:3]) or "Evidence that margins, growth or cash conversion move materially away from the assumptions above."))
        memo = (f"# {info.get('shortName', sym)} ({sym}) research memo\n\n**Overall verdict: {verdict}** (short-term {rec['short']['rating']}, long-term {rec['long']['rating']})  \nPrice ${price:,.2f} as of {stamp} (Yahoo Finance via yfinance, delayed)\n\n"
                + (f"Fair value: bear ${scen['Bear']['fv']:,.2f} / base ${scen['Base']['fv']:,.2f} / bull ${scen['Bull']['fv']:,.2f} / probability-weighted ${pw:,.2f}  \n"
                   f"Expected annualized return: 1y {pct(ret[1])}, 3y {pct(ret[3])}, 5y {pct(ret[5])}  \nMargin of safety: {pct(1 - price / pw)}\n\n" if scen else "")
                + "## Scores\n" + "\n".join(f"- {r['Score']}: {r['Value']}" for r in rows) + "\n\n## Flags\n"
                + "\n".join(f"- {f[1]}: {f[2]}" for f in flags) + "\n" + "\n".join(f"- {c}" for c in cons)
                + "\n\nEducational tool, not financial advice. Not evaluated: " + "; ".join(NOT_EVALUATED))
        st.download_button("⬇️ Download memo (.md)", memo, file_name=f"{sym}_memo.md", width="stretch")

    with tabs[1]:
        if base_dcf:
            tag = [("Revenue base", f"{rev0:,.0f}", "Reported (TTM)"), ("Starting op margin", pct(m0), "Reported (TTM)"),
                   ("Early growth", pct(g1), "Analyst estimate" if eg is not None and abs(g1 - g1_def) < 1e-9 else "Calculated / your override"),
                   ("Target margin", pct(mt), "Calculated blend of current and 4-yr avg / override"), ("WACC", f"{wacc:.1%}", f"Modeled: rf {rf:.2%} {'(live ^TNX)' if rf_live else '(fallback assumption, live rate unavailable)'}, beta {beta:.2f}, ERP 4.5% (assumption)"),
                   ("Terminal growth", f"{tg:.1%}", "Assumption"), ("Tax rate", "21%", "Assumption"), ("Sales / capital", f"{s2c:.1f}", "Assumption (sector default)"),
                   ("Net debt", f"{nd:,.0f}", "Reported"), ("Shares", f"{shares:,.0f}", "Reported (basic; dilution not modeled)")]
            st.dataframe(pd.DataFrame(tag, columns=["Input", "Value", "Type"]), hide_index=True, width="stretch")
            st.metric("DCF fair value / share", f"${base_dcf['per_share']:,.2f}", pct(base_dcf["per_share"] / price - 1))
            st.caption(f"{base_dcf['tv_share']:.0%} of enterprise value comes from the terminal value. Stock comp is treated as a cost (GAAP margins).")
            st.markdown(f"**Reverse DCF:** {'price implies ' + pct(impl) + ' early-year growth, which looks **' + plaus.lower() + '**.' if impl is not None else 'no solution between -25% and +60% growth.'}")
            st.write("Sensitivity: WACC (rows) vs terminal growth (columns)")
            st.dataframe(R.sensitivity(p, "wacc", "tg", [-.01, -.005, 0, .005, .01], [-.005, 0, .005]).round(1), width="stretch")
            st.write("Sensitivity: early growth (rows) vs target margin (columns)")
            st.dataframe(R.sensitivity(p, "g1", "mt", [-.04, -.02, 0, .02, .04], [-.04, -.02, 0, .02, .04]).round(1), width="stretch")
            with st.expander("Year-by-year projection"):
                st.dataframe(base_dcf["rows"].round(3), hide_index=True, width="stretch")
        elif pb_ok:
            st.metric("Justified price/book value", f"${scen['Base']['fv']:,.2f}", pct(scen["Base"]["fv"] / price - 1))
            st.caption("Banks and REITs are valued on (ROE - g) / (cost of equity - g) times book value per share, since cash-flow DCFs don't suit balance-sheet businesses.")
        else:
            show_missing(d, kind, extra, info, price, mult)

    with tabs[2]:
        if scen:
            st.dataframe(pd.DataFrame([{"Case": k, "Probability": f"{v['prob']:.0%} (range ±10 pts)", "Fair value": f"${v['fv']:,.2f}", "vs price": pct(v["fv"] / price - 1),
                                        **({"Growth": pct(v["p"]["g1"]), "Target margin": pct(v["p"]["mt"]), "WACC": f"{v['p']['wacc']:.1%}"} if "p" in v else {})}
                                       for k, v in scen.items()]), hide_index=True, width="stretch")
            st.caption("Probabilities start at 25/50/25 and tilt up to ±10 points with the fundamentals score (profitability, growth, balance sheet). "
                       "They are judgment, not statistics; extreme outcomes may be under-represented.")
            if p:
                st.write("Stress tests (applied to the base case)")
                s = R.stress(p, price)
                s["Fair value"] = s["Fair value"].map("${:,.2f}".format); s["vs price"] = s["vs price"].map(pct)
                st.dataframe(s, hide_index=True, width="stretch")
        else:
            show_missing(d, kind, extra, info, price, mult)

    with tabs[3]:
        st.subheader("Earnings quality & forensic checks")
        for sev, name, why in flags:
            st.write(f"{'🔴' if sev == 'high' else '🟠'} **{name}**: {why}")
        if not flags:
            st.write("🟢 No forensic flags from the available statements.")
        if fm:
            st.dataframe(pd.DataFrame([(k, f"{v:,.3f}") for k, v in fm.items()], columns=["Metric", "Value"]), hide_index=True, width="stretch")
        st.subheader("Contradiction checks")
        for c in cons:
            st.write(f"⚠️ {c}")
        if not cons:
            st.write("No contradictions detected among the inputs that were available.")
        st.caption("Flags are reasons to investigate, not evidence of misconduct.")

    with tabs[4]:
        qs = R.quant(hist["Close"], extra["spy"], rf)
        if qs:
            st.dataframe(pd.DataFrame([(k, f"{v:.3f}") for k, v in qs.items()], columns=["Statistic", "Value"]), hide_index=True, width="stretch")
            st.caption("Backward-looking; not predictive. Beta, alpha and relative strength are measured against SPY.")
        pe = load_peers(sym, info.get("industryKey")) if info.get("industryKey") else None
        if pe is not None and len(pe):
            st.write("Industry peers (Yahoo's top companies in the same industry)")
            st.dataframe(pe.round(3), hide_index=True, width="stretch")
            for lab, key in (("Forward P/E", "forwardPE"), ("EV/EBITDA", "enterpriseToEbitda"), ("P/S", "priceToSalesTrailing12Months")):
                med, own = pe[lab if lab != "P/S" else "P/S"].median(), info.get(key)
                if own and med and med > 0:
                    st.write(f"{lab}: {own:.1f} vs peer median {med:.1f} ({own / med - 1:+.0%})")
        else:
            st.info("Peer data unavailable for this ticker (Yahoo industry list not returned).")

    with tabs[5]:
        t1, t2 = st.tabs(["Short-term", "Long-term"])
        with t1:
            render(short, df, price)
        with t2:
            render(long_, df, price)

    with tabs[6]:
        for x in warns:
            st.warning(x)
        if not warns:
            st.success("No data-validation warnings.")
        st.write(f"Retrieved {extra['fetched']:%Y-%m-%d %H:%M} UTC.")
        st.dataframe(pd.DataFrame(extra["sources"], columns=["Item", "Source", "Detail"]), hide_index=True, width="stretch")
        cv = coverage(df, profile_for(info, sector))
        st.markdown(f"**Metric coverage:** {len(cv['evaluated'])} of {cv['total']} quantitative metrics evaluated · {len(cv['unavailable'])} unavailable "
                    f"(data not reported or too little history) · {len(cv['excluded'])} not applicable to this sector. "
                    f"Qualitative rubric: {sum(1 for i in qitems.values() if i['score'] is not None)} of {len(qitems)} items assessed.")
        with st.expander("Which metrics were unavailable or excluded"):
            st.write("Unavailable: " + (", ".join(cv["unavailable"]) or "none"))
            st.write("Not applicable to this sector: " + (", ".join(cv["excluded"]) or "none"))
        st.write("**Not evaluated (data or analysis not available here):**")
        for n in NOT_EVALUATED:
            st.write(f"• {n}")
        for x in d["issues"]:
            st.warning(x)

    with tabs[7]:
        st.write("Self-checks recomputed from the same inputs the verdict used. A ⚠️ means investigate before trusting the number.")
        ck = R.reconcile(info, price, shares, nd, mi, base_dcf, scen)
        st.dataframe(pd.DataFrame([{"Check": a_, "Result": "✅" if ok else "⚠️", "Detail": d_} for a_, ok, d_ in ck]), hide_index=True, width="stretch")
        viol = R.check_consistency(rec)
        st.success("Rating consistency: short, long and overall are mutually consistent.") if not viol else [st.error(v_) for v_ in viol]
        rows_ = [("Price", price), ("Pure DCF base case", base_dcf["per_share"] if base_dcf else None),
                 ("Multiples base case (fwd EPS x sector P/E)", mult["Base"]["fv"] if mult else None),
                 ("Blended base case used", scen["Base"]["fv"] if scen else None), ("Probability-weighted", pw)]
        st.dataframe(pd.DataFrame([(n_, "n/a" if v_ is None else f"${v_:,.2f}") for n_, v_ in rows_], columns=["Valuation", "Per share"]), hide_index=True, width="stretch")
        st.write(f"DCF vs multiples disagreement: **{disagree:.0%}** " + ("(range widened to span both methods; strong ratings disabled)" if disagree > .5 else ""))
        nn = lambda x: "n/a" if x is None else f"{x:,.0f}"
        st.write(f"Net debt used: {nn(nd)} ({d['nd_note']}); minority interest {nn(mi)}; shares {nn(shares)} (= market cap / price); "
                 f"beta raw {info.get('beta')} -> shrunk {beta:.2f}; risk-free {rf:.2%} {'live' if rf_live else 'fallback'}.")
        worst = sorted(contrib.items(), key=lambda kv: kv[1])[:3]
        st.write("Biggest drags on the overall score: " + ", ".join(f"{k} ({v:+.2f})" for k, v in worst))

    with tabs[8]:
        st.write(f"Rubric coverage **{qcov:.0%}**: {sum(1 for i in qitems.values() if i['score'] is not None)} of {len(qitems)} questions have evidence. "
                 "Eight can be computed from reported statements; six need business knowledge or filing text.")
        st.dataframe(pd.DataFrame([{"Question": i["label"], "Score": "not assessed" if i["score"] is None else f"{i['score']:+.2f}",
                                    "Confidence": f"{i['conf']:.0%}", "Evidence": i["evidence"], "Source": i["source"]} for i in qitems.values()]),
                     hide_index=True, width="stretch")
        key = _secret("ANTHROPIC_API_KEY")
        if not key:
            st.info("Optional AI analyst: add ANTHROPIC_API_KEY to the app's secrets to have an AI model assess competitive position, industry structure, management, "
                    "AI value capture, customer/regulatory risk and disruption, and challenge the numbers. It uses a paid API; the rest of the app is free. "
                    "Without it these questions stay 'not assessed' and reduce qualitative coverage.")
        elif st.button("Run AI analyst", width="stretch", key=f"runai{sym}"):
            facts = {"ticker": sym, "sector": sector, "industry": info.get("industry"), "price": price, "forward_pe": info.get("forwardPE"),
                     "revenue_growth_yoy": info.get("revenueGrowth"), "operating_margin": m0, "gross_margin": info.get("grossMargins"),
                     "historical_revenue_cagr": hist_g, "price_implies_early_growth": impl, "reverse_dcf_assessment": plaus,
                     "dcf_base_value_per_share": base_dcf["per_share"] if base_dcf else None, "forensic_flags": [f[1] for f in flags],
                     "computed_rubric": {k_: (i_["score"], i_["evidence"]) for k_, i_ in qitems.items() if i_["score"] is not None}}
            with st.spinner("Asking the AI analyst..."):
                text, err = Q.call_llm(Q.llm_prompt(facts, info.get("longBusinessSummary")), key, _secret("ANTHROPIC_MODEL") or "claude-sonnet-5-5")
            if err:
                st.error(f"AI analyst unavailable: {err}")
            else:
                its, ch, errs = Q.parse_llm(text)
                st.session_state[f"ai_{sym}"] = dict(items=its, challenge=ch, errs=errs)
                st.rerun()
        if ai:
            for e_ in ai["errs"]:
                st.caption(f"AI response check: {e_}")
            if ai.get("challenge"):
                st.write(f"**AI analyst challenge ({ai['challenge']['adjustment']:+.2f}):** {ai['challenge']['claim']}  \nEvidence: {ai['challenge']['evidence']}")
                st.checkbox("Apply this challenge to the ratings (explicit override, max ±0.25)", key=f"ai_apply_{sym}")
            st.caption("AI items are unverified judgments from the model, shown with their stated basis and confidence haircut. They never replace computed evidence.")


def main():
    st.set_page_config(page_title="Stock Rater", page_icon="📈", layout="centered")
    st.title("📈 Stock Rater")
    st.caption("Type a ticker. Sector-adjusted short- and long-term ratings, with a written thesis.")

    with st.form("f"):
        sym_in = st.text_input("Ticker", placeholder="e.g. AAPL").strip().upper()
        go = st.form_submit_button("Analyze", width="stretch")
    if go and sym_in:
        st.session_state["sym"] = sym_in
    sym = st.session_state.get("sym")
    if not sym:
        return

    with st.spinner(f"Crunching {sym}..."):
        info, hist, inc, cf, bs, extra = prepare(sym)
        sector = info.get("sector")
        prof = profile_for(info, sector)
        df = build_metrics(info, hist, inc, cf, bs, prof)
    if df.empty or hist is None or hist.empty:
        st.error(f"Couldn't get enough data for '{sym}'. Check the ticker and try again.")
        return

    price = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
    vol_row = df.loc[df.Metric == "Annualized volatility", "Value"]
    vol = float(vol_row.iloc[0]) if len(vol_row) else .35
    short, long_ = analyze(df, price, vol, info), analyze_long(df, price, info, prof)
    th = thesis(df, info, short, long_, sector)

    research_ui(sym, info, hist, inc, cf, bs, df, extra, sector, price, short, long_, th)

    with st.expander(f"Quantitative metrics evaluated for this stock: {len(df)} of {len(CATALOG)} in the library"):
        show = df.assign(Signal=df.Score.map(lambda s: "🟢" if s > .3 else ("🔴" if s < -.3 else "🟡")))
        show = show[["Signal", "Category", "Metric", "Value", "Score"]].copy()
        show["Value"] = show["Value"].map(fmt)
        show["Score"] = show["Score"].round(2)
        st.dataframe(show, hide_index=True, width="stretch")

    st.caption("Educational tool, not financial advice. Based on free, sometimes incomplete Yahoo Finance "
               "data and simple rules of thumb. Metrics unavailable for a ticker (e.g. ETFs) are skipped.")


main()
