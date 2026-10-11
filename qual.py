"""Qualitative assessment. 14 rubric items. Eight are computed from reported statements (evidence is shown); six need
business knowledge or filing text and are scored only by the optional AI analyst. Unassessed items are shown as such and
reduce qualitative coverage, which in turn reduces how much the qualitative view can move a rating (it never fakes coverage)."""
import json, re
import numpy as np
import research as R

DATA_ITEMS = {"pricing_power": "Pricing power", "moat_economics": "Returns on capital vs cost of capital (moat evidence)",
              "growth_durability": "Growth durability", "margin_sustainability": "Margin sustainability",
              "capital_allocation": "Capital allocation", "incentive_alignment": "Incentives and dilution (stock comp)",
              "resilience": "Balance-sheet resilience", "narrative_vs_reality": "Narrative vs reality (what the price implies)"}
LLM_ITEMS = {"competitive_position": "Competitive position and moat type", "industry_structure": "Industry structure: who captures the value",
             "management_quality": "Management quality and track record", "ai_value_capture": "AI: value capture, not just exposure",
             "customer_and_regulatory_risk": "Customer concentration, regulatory and legal risk", "disruption_risk": "Technological disruption risk"}
ALL_ITEMS = {**DATA_ITEMS, **LLM_ITEMS}
NOT_ASSESSED = "Not assessed: needs filing text or business knowledge. Enable the optional AI analyst or form your own view."
clip = lambda x: float(np.clip(x, -1, 1))


def _row(df, names):
    return R.getrow(df, names)


def assess_data(inc, cf, bs, info, V, wacc=.09):
    """Evidence-derived items. Each returns score in [-1, 1], confidence in [0, 1], and the evidence behind it."""
    items = {}
    rev, oi, ni, gp = (_row(inc, n) for n in (["Total Revenue", "Operating Revenue"], ["Operating Income"], ["Net Income"], ["Gross Profit"]))
    eq, debt, cash = _row(bs, ["Stockholders Equity", "Common Stock Equity"]), _row(bs, ["Total Debt"]), _row(bs, ["Cash Cash Equivalents And Short Term Investments"])
    sbc, fcf, gw = _row(cf, ["Stock Based Compensation"]), _row(cf, ["Free Cash Flow"]), _row(bs, ["Goodwill"])
    sh = _row(bs, ["Ordinary Shares Number", "Share Issued"])

    def put(k, score, conf, ev):
        items[k] = dict(id=k, label=ALL_ITEMS[k], score=clip(score), conf=float(conf), evidence=ev, source="computed from financial statements")
    if gp is not None and rev is not None and len(gp) >= 3:
        gm = (gp / rev).dropna()
        lvl, sd = float(gm.iloc[0]), float(gm.std())
        put("pricing_power", clip((lvl - .35) / .25) * .7 + clip((.03 - sd) / .03) * .3, .6,
            f"Gross margin {lvl:.0%} (std dev {sd:.1%} over {len(gm)} yrs). High, stable margins suggest pricing power; a proxy, not proof.")
    if oi is not None and eq is not None and debt is not None and cash is not None and len(oi) >= 3:
        ic = (eq + debt - cash).dropna()
        roic = (oi * (1 - .21) / ic).dropna()
        if len(roic) >= 3 and (ic > 0).all():
            avg = float(roic.mean())
            put("moat_economics", clip((avg - wacc) / .10), .6, f"Average ROIC proxy {avg:.0%} vs ~{wacc:.0%} cost of capital over {len(roic)} yrs. "
                "Persistent excess returns are the financial footprint of an advantage; ROIC here ignores goodwill-adjusted capital.")
    if rev is not None and len(rev) >= 4:
        gr = (rev.iloc[:-1].values / rev.iloc[1:].values - 1)
        share_up, cagr, last = float((gr > 0).mean()), float((rev.iloc[0] / rev.iloc[-1]) ** (1 / (len(rev) - 1)) - 1), float(gr[0])
        put("growth_durability", clip(share_up - .5) * .8 + clip((last - cagr) / .10) * .2 + clip(cagr / .15) * .4 - .2, .5,
            f"Revenue grew in {share_up:.0%} of years, {cagr:.0%} CAGR, latest year {last:.0%} ({'accelerating' if last > cagr else 'decelerating'}).")
    if oi is not None and rev is not None and len(oi) >= 3:
        m = (oi / rev).dropna()
        put("margin_sustainability", clip((float(m.mean()) - float(m.iloc[0])) / .05 + .2 if m.iloc[0] > m.mean() * 1.15 else .3 if m.iloc[0] >= m.mean() else -.2), .5,
            f"Operating margin {m.iloc[0]:.0%} now vs {m.mean():.0%} average: " + ("above history, so peak-margin risk." if m.iloc[0] > m.mean() * 1.15 else "in line with or below history."))
    parts, ev = [], []
    if sh is not None and len(sh) >= 2:
        ch = float(sh.iloc[0] / sh.iloc[-1] - 1) / max(len(sh) - 1, 1); parts.append(clip(-ch / .02)); ev.append(f"share count {ch:+.1%}/yr")
    if gw is not None and eq is not None and eq.iloc[0] > 0:
        r = float(gw.iloc[0] / eq.iloc[0]); parts.append(clip((.4 - r) / .4)); ev.append(f"goodwill {r:.0%} of equity (acquisition history)")
    if fcf is not None and sbc is not None:
        v = float(fcf.iloc[0] - sbc.iloc[0]); parts.append(1.0 if v > 0 else -1.0); ev.append(f"FCF after stock comp {'positive' if v > 0 else 'negative'}")
    if parts:
        put("capital_allocation", float(np.mean(parts)), .4, "; ".join(ev) + ". Buyback price discipline and acquisition returns are not visible in these figures.")
    if sbc is not None and rev is not None:
        r = float(sbc.iloc[0] / rev.iloc[0]); put("incentive_alignment", clip((.05 - r) / .06), .4, f"Stock comp is {r:.1%} of revenue (heavy comp transfers value from shareholders).")
    if debt is not None and cash is not None and oi is not None:
        nd = float(debt.iloc[0] - cash.iloc[0]); ebit = float(oi.iloc[0])
        lev = nd / ebit if ebit > 0 else 9.0
        put("resilience", clip((2.5 - lev) / 3), .6, f"Net debt {'(net cash)' if nd < 0 else f'= {lev:.1f}x operating income'}.")
    if V and V.get("plaus") not in (None, "n/a") and V.get("impl") is not None:
        s = {"Conservative": .8, "Reasonable": .3, "Aggressive": -.5, "Implausible": -1.0}[V["plaus"]]
        put("narrative_vs_reality", s, .6, f"The price implies {V['impl']:+.1%} early-year growth, which looks {V['plaus'].lower()} against the company's own history.")
    for k, label in ALL_ITEMS.items():
        if k not in items:
            items[k] = dict(id=k, label=label, score=None, conf=0.0, evidence=NOT_ASSESSED if k in LLM_ITEMS else "Insufficient reported data for this item.", source="none")
    return items


def summarize(items):
    """(score in [-1, 1], coverage in [0, 1]). Coverage = share of the 14 rubric items with evidence, confidence-weighted."""
    got = [i for i in items.values() if i["score"] is not None and i["conf"] > 0]
    if not got:
        return 0.0, 0.0
    w = sum(i["conf"] for i in got)
    return float(sum(i["score"] * i["conf"] for i in got) / w), float(sum(i["conf"] for i in got) / len(ALL_ITEMS))


# ------------------------------------------------------------------ optional AI analyst
def llm_prompt(facts, summary):
    return ("You are a skeptical fundamental equity analyst. Use ONLY the FACTS and BUSINESS SUMMARY below plus well-established general "
            "knowledge. If you do not know, set score to null. Never invent numbers, customers, or events. Judge value capture and durability, not hype.\n\n"
            f"FACTS (reported or calculated, with sources):\n{json.dumps(facts, default=str)[:6000]}\n\nBUSINESS SUMMARY:\n{(summary or 'not available')[:2500]}\n\n"
            "Score each item from -1 (very negative for shareholders) to +1 (very positive): " + ", ".join(f"{k} ({v})" for k, v in LLM_ITEMS.items()) +
            ".\nReturn ONLY JSON: {\"items\":[{\"id\":str,\"score\":number|null,\"confidence\":0-1,\"evidence\":str(<=300 chars),"
            "\"basis\":\"provided_facts|business_summary|general_knowledge\"}],\"challenge\":null|{\"claim\":str,\"evidence\":str,"
            "\"adjustment\":number(-0.25..0.25)}}. Use challenge only if the numbers above are misleading for a specific, evidenced reason.")


def parse_llm(text):
    """Validates the model's JSON. Returns (items dict, challenge or None, errors). Invalid pieces are dropped, never repaired."""
    errs, items, challenge = [], {}, None
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        data = json.loads(m.group(0))
    except Exception as e:
        return {}, None, [f"Unparseable AI response: {e}"]
    for it in data.get("items", []) if isinstance(data, dict) else []:
        k = it.get("id") if isinstance(it, dict) else None
        if k not in LLM_ITEMS:
            errs.append(f"Unknown item id {k!r} ignored"); continue
        sc, cf_, ev = it.get("score"), it.get("confidence", 0), str(it.get("evidence", ""))[:300]
        basis = it.get("basis") if it.get("basis") in ("provided_facts", "business_summary", "general_knowledge") else "general_knowledge"
        if sc is None or not isinstance(sc, (int, float)) or not ev.strip():
            errs.append(f"{k}: no usable score or evidence, left unassessed"); continue
        conf = float(np.clip(cf_ if isinstance(cf_, (int, float)) else 0, 0, 1)) * (.6 if basis == "general_knowledge" else .8)
        items[k] = dict(id=k, label=LLM_ITEMS[k], score=clip(sc), conf=conf, evidence=ev, source=f"AI analyst ({basis}); unverified")
    ch = data.get("challenge") if isinstance(data, dict) else None
    if isinstance(ch, dict) and str(ch.get("claim", "")).strip() and str(ch.get("evidence", "")).strip() and isinstance(ch.get("adjustment"), (int, float)):
        challenge = dict(claim=str(ch["claim"])[:300], evidence=str(ch["evidence"])[:400], adjustment=float(np.clip(ch["adjustment"], -.25, .25)))
    elif ch:
        errs.append("Challenge ignored: needs claim, evidence and a numeric adjustment")
    return items, challenge, errs


def call_llm(prompt, key, model="claude-sonnet-5-5", timeout=60):
    try:
        import requests
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=timeout,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": model, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return "".join(b.get("text", "") for b in r.json().get("content", [])), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
