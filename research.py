"""Deterministic research engine. No UI, no network: every number here is reproducible from its inputs."""
import numpy as np
import pandas as pd

DEFAULT_W = {"Business quality": .15, "Growth quality": .10, "Valuation attractiveness": .25,
             "Expected return": .25, "Downside risk": .15, "Financial strength": .10}


def getrow(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s):
                return s
    return None


# ---------------------------------------------------------------- valuation
def dcf(rev0, g1, m0, mt, wacc, tg, tax, s2c, net_debt, shares, years=10, hold=3):
    """FCFF DCF. Growth held at g1 for `hold` years, then fades linearly to tg; operating margin moves linearly m0 -> mt.
    Reinvestment = change in revenue / sales-to-capital. EBIT is GAAP, so stock comp is already a cost."""
    wacc = max(wacc, tg + .01)
    rev, pv, rows = rev0, 0.0, []
    for t in range(1, years + 1):
        g = g1 if t <= hold else g1 + (tg - g1) * (t - hold) / (years - hold)
        new = rev * (1 + g)
        m = m0 + (mt - m0) * t / years
        ebit = new * m
        fcff = (ebit * (1 - tax) if ebit > 0 else ebit) - (new - rev) / s2c
        pv += fcff / (1 + wacc) ** t
        rows.append({"Year": t, "Revenue": new, "Growth": g, "Op margin": m, "FCFF": fcff})
        rev = new
    roic = max(mt * (1 - tax) * s2c, wacc)  # terminal growth never creates value below the cost of capital
    nopat_t = max(rev * (1 + tg) * mt, 0) * (1 - tax)
    pv_tv = nopat_t * (1 - tg / roic) / (wacc - tg) / (1 + wacc) ** years
    ev = pv + pv_tv
    return {"per_share": (ev - net_debt) / shares, "ev": ev,
            "tv_share": pv_tv / ev if ev > 0 else float("nan"), "rows": pd.DataFrame(rows)}


def reverse_dcf(price, **p):
    """Growth rate that makes the DCF equal today's price (None if no solution in -25%..60%)."""
    f = lambda g: dcf(g1=g, **p)["per_share"] - price
    lo, hi = -.25, .60
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
    return (lo + hi) / 2


def plausibility(g, hist_g):
    if g is None:
        return "n/a"
    if hist_g is not None and hist_g > .02:
        r = g / hist_g
        return "Conservative" if r <= .5 else "Reasonable" if r <= 1.0 else "Aggressive" if r <= 1.5 else "Implausible"
    return "Conservative" if g <= .03 else "Reasonable" if g <= .08 else "Aggressive" if g <= .15 else "Implausible"


def probs(q):
    """Scenario probabilities tilted by fundamentals score q in [-1, 1]; always sum to 1."""
    pb = float(np.clip(.25 + .10 * q, .10, .40))
    pr = float(np.clip(.25 - .10 * q, .10, .40))
    return {"Bull": pb, "Base": 1 - pb - pr, "Bear": pr}


def scenarios(p, q):
    """Cash-flow scenarios with symmetric deltas around base. The discount rate is held constant so risk is not
    counted twice (in cash flows AND in WACC); rate shocks are handled separately in stress()."""
    dg, dm = max(.03, .25 * abs(p["g1"])), max(.03, .15 * abs(p["mt"]))
    bull = {**p, "g1": p["g1"] + dg, "mt": p["mt"] + dm, "tg": p["tg"] + .0025}
    bear = {**p, "g1": p["g1"] - dg, "mt": p["mt"] - dm, "tg": max(p["tg"] - .0025, 0)}
    pr = probs(q)
    return {k: {"fv": max(dcf(**v)["per_share"], 0.0), "prob": pr[k], "p": v}
            for k, v in (("Bull", bull), ("Base", p), ("Bear", bear))}


def stress(p, price):
    cases = {"Recession (growth -8 pts, margin -25%)": {"g1": p["g1"] - .08, "mt": p["mt"] * .75 if p["mt"] > 0 else p["mt"]},
             "Rates +2 pts": {"wacc": p["wacc"] + .02},
             "Margin compression (-30%)": {"mt": p["mt"] * .7 if p["mt"] > 0 else p["mt"]},
             "Terminal growth -1 pt": {"tg": max(p["tg"] - .01, 0)}}
    return pd.DataFrame([{"Stress": k, "Fair value": max(dcf(**{**p, **v})["per_share"], 0),
                          "vs price": max(dcf(**{**p, **v})["per_share"], 0) / price - 1} for k, v in cases.items()])


def sensitivity(p, a, b, da, db):
    """Grid of fair value per share varying parameters a and b by offsets da x db."""
    return pd.DataFrame({f"{b} {p[b] + y:.3f}": [dcf(**{**p, a: p[a] + x, b: p[b] + y})["per_share"] for x in da]
                         for y in db}, index=[f"{a} {p[a] + x:.3f}" for x in da])


def fin_scenarios(bvps, roe, ke, g, q):
    """Justified price/book = (ROE - g) / (ke - g), for banks and similar balance-sheet businesses."""
    f = lambda r, k: max(bvps * (r - g) / (max(k, g + .01) - g), 0.0)
    pr = probs(q)
    return {"Bull": {"fv": f(roe * 1.1, ke - .01), "prob": pr["Bull"]}, "Base": {"fv": f(roe, ke), "prob": pr["Base"]},
            "Bear": {"fv": f(roe * .75, ke + .01), "prob": pr["Bear"]}}


def expected_returns(price, fv, ke, convergence_years=3):
    """Annualized return if price converges to fair value over `convergence_years`, with value compounding at ke."""
    out = {}
    for t in (1, 3, 5):
        w = min(t / convergence_years, 1.0)
        pt = price * (1 + ke) ** t * (max(fv, 1e-9) / price) ** w  # value compounds at ke; the gap closes gradually
        out[t] = (pt / price) ** (1 / t) - 1
    return out


# ---------------------------------------------------------------- forensic & logic checks
def forensic(inc, cf, bs):
    flags, m = [], {}
    g = lambda df, n: getrow(df, n)
    ni, rev = g(inc, ["Net Income"]), g(inc, ["Total Revenue", "Operating Revenue"])
    cfo, fcf, sbc = g(cf, ["Operating Cash Flow"]), g(cf, ["Free Cash Flow"]), g(cf, ["Stock Based Compensation"])
    ar, inv = g(bs, ["Accounts Receivable", "Receivables"]), g(bs, ["Inventory"])
    gw, eq, ta = g(bs, ["Goodwill"]), g(bs, ["Stockholders Equity", "Common Stock Equity"]), g(bs, ["Total Assets"])
    sh = g(bs, ["Ordinary Shares Number", "Share Issued"])
    gr = lambda s: s.iloc[0] / s.iloc[1] - 1 if s is not None and len(s) > 1 and s.iloc[1] > 0 else None
    if ni is not None and cfo is not None and ni.iloc[0] > 0:
        m["Cash conversion (CFO / net income)"] = cfo.iloc[0] / ni.iloc[0]
        if m["Cash conversion (CFO / net income)"] < .8:
            flags.append(("high", "Weak cash conversion", "Operating cash flow is well below reported earnings; investigate accruals."))
    if ni is not None and cfo is not None and ta is not None and ta.iloc[0] > 0:
        m["Accruals / assets"] = (ni.iloc[0] - cfo.iloc[0]) / ta.iloc[0]
        if m["Accruals / assets"] > .10:
            flags.append(("high", "High accruals", "Earnings rely heavily on non-cash items."))
    for nm, s in (("Receivables", ar), ("Inventory", inv)):
        if gr(s) is not None and gr(rev) is not None:
            m[f"{nm} growth minus revenue growth"] = gr(s) - gr(rev)
            if gr(s) - gr(rev) > .15:
                flags.append(("med", f"{nm} outpacing sales", "Can signal weak demand, channel stuffing or aggressive recognition (a reason to investigate, not proof)."))
    if sbc is not None and rev is not None:
        m["Stock comp / revenue"] = sbc.iloc[0] / rev.iloc[0]
        if fcf is not None:
            m["FCF after stock comp"] = fcf.iloc[0] - sbc.iloc[0]
        if m["Stock comp / revenue"] > .10:
            flags.append(("med", "Heavy stock-based compensation", "Reported free cash flow overstates what shareholders keep."))
    if gw is not None and eq is not None and eq.iloc[0] > 0:
        m["Goodwill / equity"] = gw.iloc[0] / eq.iloc[0]
        if m["Goodwill / equity"] > .5:
            flags.append(("med", "Goodwill is large vs equity", "Impairment risk if acquisitions underperform."))
    if sh is not None and len(sh) > 1:
        m["Share count change (YoY)"] = sh.iloc[0] / sh.iloc[1] - 1
        if m["Share count change (YoY)"] > .03:
            flags.append(("med", "Meaningful dilution", "Per-share value grows slower than the business."))
    return flags, m


def contradictions(c):
    """c: dict of optional numbers. Each rule fires only when every input it needs is present."""
    has = lambda *k: all(c.get(x) is not None for x in k)
    rules = [
        (has("rev_g", "margin_chg") and c["rev_g"] > .15 and c["margin_chg"] < -.02, "High revenue growth but operating margins are shrinking: growth may be bought, not earned."),
        (has("eps_g", "oi_g", "share_chg") and c["eps_g"] > .10 and c["oi_g"] < .03 and c["share_chg"] < -.02, "EPS growth is mostly buybacks; operating income is flat."),
        (has("cash_conv") and c["cash_conv"] < .7, "Reported earnings are not converting to cash."),
        (has("fwd_pe", "margin_peak") and c["fwd_pe"] < 12 and c["margin_peak"] and c["cyclical"], "Cheap forward P/E in a cyclical sector at peak margins: earnings may be unsustainable."),
        (has("ps", "gross_m") and c["ps"] < 2 and c["gross_m"] < .25, "Low P/S is not cheap when gross margins are structurally thin."),
        (has("fcf_yield", "rev_g") and c["fcf_yield"] > .06 and c["rev_g"] < 0, "High FCF yield alongside shrinking sales: the yield may be a melting ice cube."),
        (has("tv_share") and c["tv_share"] > .75, "Over 75% of DCF value sits in the terminal value: the result is very assumption-sensitive."),
        (has("quality", "exp5", "ke") and c["quality"] > .4 and c["exp5"] < c["ke"], "Excellent business, but the modeled return at today's price is below the required return."),
        (has("analyst", "rev_trend") and c["analyst"] > .5 and c["rev_trend"] < -.02, "Analysts are bullish while their own EPS estimates are being cut."),
    ]
    return [txt for ok, txt in rules if ok]


def validate(info, hist, price, now):
    w = []
    if hist is None or hist.empty:
        return ["No price history returned."]
    age = (now - hist.index[-1].tz_localize(None)).days if hist.index[-1].tzinfo else (now - hist.index[-1]).days
    if age > 5:
        w.append(f"Latest price bar is {age} days old (stale or market closed for a long period).")
    mc, sh = info.get("marketCap"), info.get("sharesOutstanding")
    if mc and sh and abs(mc / (price * sh) - 1) > .15:
        w.append("Market cap and shares x price disagree by >15% (multiple share classes or stale data).")
    if not info.get("sector"):
        w.append("No sector reported; generic thresholds used and industry logic limited.")
    return w


# ---------------------------------------------------------------- quant
def _naive(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
    return s


def quant(c, spy, rf=.04):
    c = _naive(c.dropna())
    if len(c) < 250:
        return {}
    r = c.pct_change().dropna()
    cagr = (c.iloc[-1] / c.iloc[0]) ** (252 / len(c)) - 1
    vol = r.std() * np.sqrt(252)
    dd = r[r < 0].std() * np.sqrt(252)
    out = {"Annualized return (history)": cagr, "Volatility": vol, "Max drawdown": (c / c.cummax() - 1).min(),
           "Downside deviation": dd, "Sharpe": (cagr - rf) / vol, "Sortino": (cagr - rf) / dd if dd else np.nan}
    if len(c) > 253:
        out["Momentum 12-1 months"] = c.iloc[-22] / c.iloc[-253] - 1
    if spy is not None and len(spy) > 250:
        s = _naive(spy["Close"].dropna()).pct_change().dropna()
        j = pd.concat([r, s], axis=1, join="inner").dropna()
        j.columns = ["a", "m"]
        if len(j) > 200:
            beta = j.cov().loc["a", "m"] / j.cov().loc["m", "m"]
            out.update({"Beta vs SPY": beta, "Correlation vs SPY": j.corr().loc["a", "m"],
                        "Alpha (annual, vs SPY)": (j.a.mean() - beta * j.m.mean()) * 252,
                        "6-mo relative strength vs SPY": (1 + j.a.tail(126)).prod() / (1 + j.m.tail(126)).prod() - 1})
    return out


# ---------------------------------------------------------------- scoring
def make_scores(cs, val, nmetrics, nflags, disagree=0.0):
    g = lambda k: float(cs.get(k, 0.0))
    clip = lambda x: float(np.clip(x, -1, 1))
    sc = {"Business quality": clip((g("Profitability") + g("Track record")) / 2),
          "Growth quality": clip(g("Growth") - .1 * min(nflags, 3)),
          "Financial strength": clip(g("Financial health")),
          "AI opportunity": None, "AI disruption risk": None}
    if val:
        rd = {"Conservative": 1, "Reasonable": .3, "Aggressive": -.5, "Implausible": -1}.get(val["plaus"], 0)
        sc["Valuation attractiveness"] = clip(.5 * clip(val["upside"] / .5) + .2 * g("Valuation") + .3 * rd)
        sc["Expected return"] = clip((val["exp5"] - val["ke"]) / .15)
        dd = max(0.0, 1 - val["bear"] / val["price"])
        sc["Downside risk"] = clip(.5 * (1 - 3 * dd) + .5 * g("Risk"))
        spread = (val["bull"] - val["bear"]) / max(val["base"], 1e-9)
    else:
        sc.update({"Valuation attractiveness": clip(g("Valuation")), "Expected return": None, "Downside risk": clip(g("Risk"))})
        spread = 1.0
    sc["Forecast confidence"] = float(np.clip(.4 * min(nmetrics / 45, 1) + .4 * (1 - min(spread / 2, 1)) + .2 * (1 - min(nflags, 5) / 5) - (.2 if disagree > .5 else 0), 0, 1))
    return sc


def overall(sc, w):
    use = {k: v for k, v in w.items() if sc.get(k) is not None and v > 0}
    tot = sum(use.values())
    contrib = {k: sc[k] * v / tot for k, v in use.items()} if tot else {}
    return sum(contrib.values()), contrib


def verdict(score, exp5, hurdle, conf=1.0, disagree=0.0):
    """Thresholds: >+0.40 Strong Buy, >+0.15 Buy, >-0.15 Hold, >-0.40 Avoid, else Strong Avoid.
    Buy ratings are capped at Hold when the modeled 5-yr return is below the required return.
    Strong ratings need evidence quality: not allowed when confidence is low or valuation methods disagree."""
    v = ("Strong Buy" if score > .40 else "Buy" if score > .15 else "Hold" if score > -.15
         else "Avoid" if score > -.40 else "Strong Avoid")
    note = ""
    if exp5 is not None and exp5 < hurdle and v in ("Strong Buy", "Buy"):
        v, note = "Hold", f"Capped at Hold: modeled 5-yr return ({exp5:.1%}) is below the {hurdle:.1%} required return."
    if v in ("Strong Buy", "Strong Avoid") and (conf < .5 or disagree > .5):
        v = "Buy" if v == "Strong Buy" else "Avoid"
        note += " Strong ratings need agreement between valuation methods and adequate confidence."
    return v, note.strip()


# ---------------------------------------------------------------- inputs & orchestration
def shrink_beta(b):
    """Blume adjustment: raw betas mean-revert toward 1."""
    return .67 * (1.0 if b is None else b) + .33


def margin_target(m0, avg, cyclical, growth=None):
    """Non-cyclical firms keep current margin if it is at/above history (no reversion below it); depressed margins
    recover 40% of the way to history. Cyclicals revert half-way to the mid-cycle average."""
    if m0 is None:
        return None
    avg = m0 if avg is None else avg
    if growth is not None and growth < 0 and not cyclical:
        avg = min(avg, m0)  # a shrinking business is not assumed to recover its historical margin
    t = .5 * m0 + .5 * avg if cyclical else (m0 if m0 >= avg else .6 * m0 + .4 * avg)
    return float(np.clip(t, -.10, .45))


def net_debt(info, bs):
    """(net debt excluding minority interest, minority interest, note). For US filers operating-lease liabilities are
    excluded from debt because EBIT already bears the lease cost (including them double-counts the burden)."""
    debt = getrow(bs, ["Total Debt"])
    cash = getrow(bs, ["Cash Cash Equivalents And Short Term Investments"])
    lease, mi = getrow(bs, ["Capital Lease Obligations"]), getrow(bs, ["Minority Interest"])
    d = float(debt.iloc[0]) if debt is not None else float(info.get("totalDebt") or 0)
    note = "balance sheet" if debt is not None else "Yahoo info (may include leases)"
    if debt is not None and lease is not None and info.get("country") == "United States":
        d -= float(lease.iloc[0])
        note += ", US operating leases excluded"
    c = float(cash.iloc[0]) if cash is not None else float(info.get("totalCash") or 0)
    return d - c, (max(float(mi.iloc[0]), 0.0) if mi is not None else 0.0), note


def use_book_value(info, sector):
    """Book-value (justified P/B) valuation is for balance-sheet businesses only: banks, insurers, lenders, REITs.
    Asset-light financials (payment networks, exchanges, asset managers) are valued on cash flows."""
    ind = (info.get("industry") or "")
    if sector == "Real Estate" or ind.startswith(("Banks", "Insurance", "Mortgage", "Thrifts")):
        return True
    if ind == "Credit Services":
        return (info.get("totalDebt") or 0) > (info.get("marketCap") or 1e18)  # lenders carry debt > equity value
    return sector == "Financial Services" and not ind


def derive(info, price, inc, bs, sector, rf_raw, est_growth, cyclical, s2c_map, mature_m=None):
    """Default valuation inputs, all deterministic and each traceable to a reported or calculated figure."""
    rev, oi = getrow(inc, ["Total Revenue", "Operating Revenue"]), getrow(inc, ["Operating Income"])
    d = {"issues": []}
    d["rev0"] = info.get("totalRevenue") or (float(rev.iloc[0]) if rev is not None else None)
    m0 = info.get("operatingMargins")
    if m0 is None and oi is not None and rev is not None:
        m0 = float(oi.iloc[0] / rev.iloc[0])
    d["m0"] = m0
    mc = info.get("marketCap") or 0
    d["shares"] = (mc / price) or info.get("sharesOutstanding")  # consistent with price and EV; handles share classes
    d["nd"], d["mi"], d["nd_note"] = net_debt(info, bs)
    d["hist_g"] = None
    if rev is not None and len(rev) >= 4 and rev.iloc[3] > 0:
        d["hist_g"] = float((rev.iloc[0] / rev.iloc[3]) ** (1 / 3) - 1)
    elif info.get("revenueGrowth") is not None:
        d["hist_g"] = info["revenueGrowth"]
    d["avg_m"] = float((oi / rev).dropna().mean()) if oi is not None and rev is not None else m0
    rf = rf_raw / 10 if rf_raw and rf_raw > .15 else rf_raw  # ^TNX is quoted in different scales over time
    d["rf_live"] = bool(rf and .005 < rf < .15)
    d["rf"] = rf if d["rf_live"] else .042
    d["beta"] = float(np.clip(shrink_beta(info.get("beta")), .5, 2.5))
    d["ke"] = d["rf"] + d["beta"] * .045
    dbt = info.get("totalDebt") or 0
    wd = dbt / (dbt + mc) if dbt + mc else 0
    d["wacc"] = float(np.clip((1 - wd) * d["ke"] + wd * (d["rf"] + .02) * (1 - .21), .065, .14))
    d["eg"] = est_growth
    ests = [x for x in (est_growth, d["hist_g"]) if x is not None]
    d["g1"] = float(np.clip(sum(ests) / len(ests) if ests else .05, -.05, .35))
    d["mt"] = margin_target(m0, d["avg_m"], cyclical, d["hist_g"])
    d["speculative"] = False
    if m0 is not None and m0 < 0:  # loss-making: holding a negative margin forever would value every grower at ~zero
        d["mt"] = float(np.clip(mature_m if mature_m is not None else .12, .05, .30)); d["speculative"] = True
        d["issues"].append("Unprofitable today: target margin assumes a mature sector-typical margin (speculative assumption).")
    fin, cur = info.get("financialCurrency"), info.get("currency")
    cur_ok = d["cur_ok"] = not (fin and cur and fin != cur)
    if not cur_ok:
        d["issues"].append(f"Statements are in {fin} but the price is in {cur}: DCF disabled (no FX conversion applied).")
    d["dcf_ok"] = bool(d["rev0"] and m0 is not None and d["shares"] and d["mt"] is not None and cur_ok)
    d["missing"] = [k for k, v in (("Revenue (TTM or latest fiscal year)", d["rev0"]), ("Operating margin", m0),
                                   ("Shares outstanding (market cap / price)", d["shares"]), ("Target operating margin", d["mt"])) if v in (None, 0)]
    d["s2c"] = s2c_map.get(sector, 1.5)
    d["tg"] = float(np.clip(d["rf"] * .6, .02, .03))
    return d


def multiples_fv(eps, lo, hi, q):
    """Cross-check: forward EPS x sector-typical fair P/E range (bear/base/bull = low/mid/high)."""
    if not eps or eps <= 0:
        return None
    pr = probs(q)
    return {"Bear": {"fv": eps * lo, "prob": pr["Bear"]}, "Base": {"fv": eps * (lo + hi) / 2, "prob": pr["Base"]},
            "Bull": {"fv": eps * hi, "prob": pr["Bull"]}}


def blend(d, m, w, q):
    """Weighted blend of DCF (weight w) and multiples. If they disagree by >50%, the range is widened to span both."""
    if d and m:
        out = {k: {**d[k], "fv": w * d[k]["fv"] + (1 - w) * m[k]["fv"]} for k in d}
        lo_, hi_ = sorted((d["Base"]["fv"], m["Base"]["fv"]))
        gap = hi_ / max(lo_, 1e-9) - 1
        if gap > .5:
            out["Bear"]["fv"] = min(out["Bear"]["fv"], d["Bear"]["fv"], m["Bear"]["fv"])
            out["Bull"]["fv"] = max(out["Bull"]["fv"], d["Bull"]["fv"], m["Bull"]["fv"])
        return out, gap
    return (d or m), 0.0


def valuate(i):
    price, q, ke = i["price"], i["q"], i["ke"]
    out = dict(scen=None, impl=None, plaus="n/a", base_dcf=None, p=None, ret=None, val=None, disagree=0.0, mult=None, pw=None, dcf_scen=None)
    d = None
    if i.get("p"):
        p = out["p"] = i["p"]
        out["base_dcf"] = dcf(**p)
        out["impl"] = reverse_dcf(price, **{k: v for k, v in p.items() if k != "g1"})
        out["plaus"] = plausibility(out["impl"], i.get("hist_g"))
        d = scenarios(p, q)
    elif i.get("pb"):
        d = fin_scenarios(i["pb"]["bvps"], i["pb"]["roe"], ke, i["pb"]["g"], q)
    m = None if i.get("pb") else multiples_fv(i.get("fwd_eps"), *i["pe_rng"], q)
    out["mult"], out["dcf_scen"] = m, d
    scen, out["disagree"] = blend(d, m, i.get("dcf_w", .6), q)
    if scen:
        out["scen"] = scen
        out["pw"] = sum(v["prob"] * v["fv"] for v in scen.values())
        out["ret"] = expected_returns(price, out["pw"], ke)
        out["val"] = dict(price=price, upside=out["pw"] / price - 1, exp5=out["ret"][5], ke=ke, bear=scen["Bear"]["fv"],
                          base=scen["Base"]["fv"], bull=scen["Bull"]["fv"], plaus=out["plaus"])
    return out


def reconcile(info, price, shares, nd, mi, base_dcf, scen):
    """Self-checks that tie the displayed numbers back to their inputs. Returns (check, passed, detail)."""
    out, mc, ev = [], info.get("marketCap"), info.get("enterpriseValue")
    if mc and shares:
        r = mc / (price * shares)
        out.append(("Market cap = price x shares", abs(r - 1) < .02, f"ratio {r:.3f}"))
    if mc and ev:
        r = (mc + nd + mi) / ev
        out.append(("Market cap + net debt (+ minority) vs Yahoo EV", abs(r - 1) < .15, f"ratio {r:.2f} (mismatch = lease/cash treatment differs)"))
    if base_dcf:
        eq = base_dcf["ev"] - nd - mi
        out.append(("DCF bridge: EV - net debt - minority = equity; / shares = per share",
                    abs(eq / shares - base_dcf["per_share"]) < 1e-6 * max(1, abs(base_dcf["per_share"])), f"${eq / shares:,.2f}"))
        out.append(("Terminal value share of DCF below 90%", base_dcf["tv_share"] < .9, f"{base_dcf['tv_share']:.0%}"))
        r = base_dcf["per_share"] / price
        out.append(("DCF within 0.5x-2x of price (else investigate inputs)", .5 <= r <= 2, f"DCF/price = {r:.2f}"))
    if scen:
        out.append(("Scenario probabilities sum to 100%", abs(sum(v["prob"] for v in scen.values()) - 1) < 1e-9, f"{sum(v['prob'] for v in scen.values()):.0%}"))
    return out


# ================================================================ THE ONE RECOMMENDATION FUNCTION
# Every rating the app displays (short-term, long-term, overall) comes from recommend(), computed once from one
# data snapshot. Nothing else in the code base is allowed to produce a Buy/Hold/Avoid label.
#
# FRAMEWORK
#  1. Scores live on [-1, +1]. One set of thresholds labels every rating:
#       > +0.40 Strong Buy | > +0.15 Buy | > -0.15 Hold | > -0.40 Avoid | else Strong Avoid
#  2. Short-term (about 3-12 months) score = 40% near-term metric model (momentum, estimate revisions, analyst view,
#     valuation, growth; plus rule-based interaction adjustments) + 35% modeled 12-month excess return + 25% downside
#     risk. "Excess return" = modeled annualized return minus the required return (cost of equity), scaled so that
#     +/-15 points of annual excess return = +/-1.
#  3. Long-term (about 3-5 years) score = the scorecard (business quality, growth quality, valuation, modeled 5-year
#     excess return, downside risk, financial strength; user-adjustable weights) + interaction adjustments.
#  4. Both horizons pass the SAME two constraints, each reported to the user with its effect:
#       - Required-return floor: no Buy-level rating if the modeled return over that horizon is below the required return.
#       - Evidence-quality gate: no Strong rating when confidence < 50% or DCF and multiples disagree by > 50%.
#     A constraint clamps the horizon SCORE to the boundary of the lower rating.
#  5. Overall score = w_short * short score + (1 - w_short) * long score (default 40/60), using the constrained scores,
#     then the same thresholds. Because the thresholds are monotonic and the overall score is a weighted average, the
#     overall rating can never fall outside the range of the two horizon ratings: Strong Buy + Strong Buy cannot
#     become Hold, and Buy + Buy cannot become Sell. check_consistency() verifies this on every analysis.
#  6. Language is never used to decide a rating. Text only explains the numbers above.
ORDER = ["Strong Avoid", "Avoid", "Hold", "Buy", "Strong Buy"]
_EPS = 1e-9


def label(s):
    return ("Strong Buy" if s > .40 else "Buy" if s > .15 else "Hold" if s > -.15 else "Avoid" if s > -.40 else "Strong Avoid")


def _constrain(score, ret, hurdle, conf, disagree, horizon):
    cons, s = [], score

    def apply(rule, detail, new):
        nonlocal s
        cons.append({"horizon": horizon, "rule": rule, "detail": detail, "from": label(s), "to": label(new)})
        s = new
    if ret is not None and ret < hurdle and s > .15:
        apply("Required-return floor", f"modeled {horizon.lower()} return {ret:.1%} is below the {hurdle:.1%} required return", .15)
    if s > .40 and (conf < .5 or disagree > .5):
        apply("Evidence-quality gate", f"confidence {conf:.0%} / DCF-vs-multiples gap {disagree:.0%} too weak for a Strong rating", .40)
    if s < -.40 and (conf < .5 or disagree > .5):
        apply("Evidence-quality gate", f"confidence {conf:.0%} / DCF-vs-multiples gap {disagree:.0%} too weak for a Strong rating", -.40 + _EPS)
    return s, cons


def recommend(cs, V, ke, metric_short, nmetrics, nflags, w=None, w_short=.40, adj_short=0.0, adj_long=0.0,
              qual=None, w_qual=.50, override=None):
    """cs: category scores; V: output of valuate(); ke: required return; metric_short: near-term metric-model composite."""
    w = w or DEFAULT_W
    val, ret, disagree = V.get("val"), V.get("ret"), V.get("disagree", 0.0)
    sc = make_scores(cs, val, nmetrics, nflags, disagree)
    card, contrib = overall(sc, w)
    conf = sc["Forecast confidence"]
    er = lambda r: None if r is None else float(np.clip((r - ke) / .15, -1, 1))
    parts = [("Near-term metric model", .40, metric_short), ("12-month excess return", .35, er(ret[1]) if ret else None),
             ("Downside risk", .25, sc["Downside risk"])]
    use = [p for p in parts if p[2] is not None]
    short_raw = float(np.clip(sum(a * b for _, a, b in use) / sum(a for _, a, _ in use) + adj_short, -1, 1))
    long_raw = float(np.clip(card + adj_long, -1, 1))
    # Qualitative blend (guiding philosophy ~50/50). The qualitative weight is scaled by how much of the rubric was actually
    # evidenced (coverage), so unassessed qualitative questions never masquerade as a neutral opinion. Short horizon: 60% of it.
    qw = {"short": 0.0, "long": 0.0}
    quant_only = {"short": short_raw, "long": long_raw}
    if qual and qual.get("coverage", 0) > 0:
        qw = {"long": w_qual * qual["coverage"], "short": .6 * w_qual * qual["coverage"]}
        short_raw = float(np.clip((1 - qw["short"]) * short_raw + qw["short"] * qual["score"], -1, 1))
        long_raw = float(np.clip((1 - qw["long"]) * long_raw + qw["long"] * qual["score"], -1, 1))
    if override:  # explicit, evidence-backed challenge to the numbers (|adjustment| <= 0.25); reported to the user
        short_raw = float(np.clip(short_raw + override["adjustment"], -1, 1)); long_raw = float(np.clip(long_raw + override["adjustment"], -1, 1))
    comps = {}
    for key, name, raw, r in (("short", "Short-term", short_raw, ret[1] if ret else None), ("long", "Long-term", long_raw, ret[5] if ret else None)):
        score, cons = _constrain(raw, r, ke, conf, disagree, name)
        comps[key] = dict(raw=raw, score=score, rating=label(score), raw_rating=label(raw), constraints=cons, ret=r)
    ov = w_short * comps["short"]["score"] + (1 - w_short) * comps["long"]["score"]
    s_, l_ = ORDER.index(comps["short"]["rating"]), ORDER.index(comps["long"]["rating"])
    trade = ""
    if abs(s_ - l_) >= 2:
        hi, lo = ("long-term", "short-term") if l_ > s_ else ("short-term", "long-term")
        trade = f" Horizon trade-off: the {hi} outlook is much stronger than the {lo} outlook; the overall weighs them {w_short:.0%}/{1 - w_short:.0%}."
    rec = dict(sc=sc, card=card, contrib=contrib, conf=conf, short=comps["short"], long=comps["long"], w_short=w_short,
               overall=dict(score=ov, rating=label(ov)), short_parts=[(n, a, b) for n, a, b in use], adj_short=adj_short, adj_long=adj_long,
               constraints=comps["short"]["constraints"] + comps["long"]["constraints"], qual_weight=qw, quant_only=quant_only,
               override=override)
    rec["summary"] = (f"Overall {rec['overall']['rating']} ({ov:+.2f}) from short-term {comps['short']['rating']} ({comps['short']['score']:+.2f}) "
                      f"and long-term {comps['long']['rating']} ({comps['long']['score']:+.2f}).{trade}")
    return rec


def check_consistency(rec):
    """Invariants that must hold for every analysis. Returns a list of violations (empty = consistent)."""
    s, l, o = (ORDER.index(rec[k]["rating"]) for k in ("short", "long", "overall"))
    out = []
    if not min(s, l) <= o <= max(s, l):
        out.append("Overall rating lies outside the range of the two horizon ratings.")
    if min(s, l) >= ORDER.index("Buy") and o < ORDER.index("Buy"):
        out.append("Both horizons are Buy or better but the overall verdict is not.")
    if max(s, l) <= ORDER.index("Avoid") and o > ORDER.index("Avoid"):
        out.append("Both horizons are Avoid or worse but the overall verdict is not.")
    if label(rec["overall"]["score"]) != rec["overall"]["rating"]:
        out.append("Overall label does not match its score.")
    return out
