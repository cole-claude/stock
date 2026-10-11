"""SEC EDGAR data layer: free, official, no API key.

Source:      data.sec.gov XBRL "companyfacts" API (+ www.sec.gov/files/company_tickers.json for ticker -> CIK).
Limits:      SEC fair-access policy = max 10 requests/second and a descriptive User-Agent that includes contact details
             (set SEC_USER_AGENT in Streamlit secrets, e.g. "MyApp me@example.com"). Requests without it may be blocked.
Licensing:   US government data, public domain. No redistribution restrictions on the facts themselves.
Coverage:    US filers of 10-K/10-Q using the us-gaap taxonomy. Foreign private issuers (20-F/40-F, IFRS) are NOT covered.
Not in it:   prices, analyst estimates, segment detail via this endpoint, industry classification (SIC is in a different API).
Freshness:   annual figures are as of the latest 10-K (can be up to ~15 months old); every series carries its filing date.
Principle:   a concept that is not reported is left missing. Nothing is estimated or substituted silently.
"""
import re
from datetime import date, datetime
import numpy as np
import pandas as pd

BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
# target -> ordered (taxonomy, concept) candidates; for each fiscal year the first candidate that has a value is used
FLOW = {
    "Total Revenue": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                      ("us-gaap", "SalesRevenueNet"), ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax")],
    "Operating Income": [("us-gaap", "OperatingIncomeLoss")],
    "Net Income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "Gross Profit": [("us-gaap", "GrossProfit")],
    "Cost Of Revenue": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold")],
    "Interest Expense": [("us-gaap", "InterestExpense"), ("us-gaap", "InterestExpenseNonoperating")],
    "Operating Cash Flow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "Capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"), ("us-gaap", "PaymentsToAcquireProductiveAssets")],
    "Stock Based Compensation": [("us-gaap", "ShareBasedCompensation")],
    "Depreciation And Amortization": [("us-gaap", "DepreciationDepletionAndAmortization"), ("us-gaap", "DepreciationAndAmortization")],
    "Diluted Average Shares": [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")],
    "Basic Average Shares": [("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")],
}
INSTANT = {
    "Total Assets": [("us-gaap", "Assets")],
    "Stockholders Equity": [("us-gaap", "StockholdersEquity")],
    "Goodwill": [("us-gaap", "Goodwill")],
    "Accounts Receivable": [("us-gaap", "AccountsReceivableNetCurrent")],
    "Inventory": [("us-gaap", "InventoryNet")],
    "Minority Interest": [("us-gaap", "MinorityInterest")],
    "Cash And ST Investments": [("us-gaap", "CashCashEquivalentsAndShortTermInvestments")],
    "Cash": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue")],
    "ST Investments": [("us-gaap", "ShortTermInvestments"), ("us-gaap", "MarketableSecuritiesCurrent")],
    "LT Debt Total": [("us-gaap", "LongTermDebt")],
    "LT Debt Noncurrent": [("us-gaap", "LongTermDebtNoncurrent")],
    "Debt Current": [("us-gaap", "LongTermDebtCurrent"), ("us-gaap", "DebtCurrent")],
    "ST Borrowings": [("us-gaap", "ShortTermBorrowings"), ("us-gaap", "CommercialPaper")],
    "Finance Lease Liability": [("us-gaap", "FinanceLeaseLiability")],
    "Shares Outstanding (FY end)": [("us-gaap", "CommonStockSharesOutstanding")],
}


def get_json(url, ua, timeout=20):
    """Returns (data, error). Never raises."""
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code} from {url.split('/')[2]}"
        return r.json(), None
    except Exception as e:  # network down, blocked, bad JSON
        return None, f"{type(e).__name__}: {e}"


def cik_for(ticker, tickers_json):
    t = ticker.upper()
    for row in (tickers_json or {}).values():
        if row.get("ticker", "").upper() == t:
            return int(row["cik_str"])
    return None


def _days(a, b):
    return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days


def annual_series(facts, candidates, instant):
    """{period_end: (value, concept, filed, form)} using annual (10-K, FY) facts; latest filing wins per period."""
    out = {}
    for tax, concept in candidates:
        node = facts.get("facts", {}).get(tax, {}).get(concept)
        if not node:
            continue
        units = node.get("units", {})
        entries = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
        best = {}
        for e in entries:
            if e.get("form") not in ("10-K", "10-K/A") or e.get("fp") != "FY":
                continue
            if not instant:
                if "start" not in e or not 340 <= _days(e["start"], e["end"]) <= 380:
                    continue
            cur = best.get(e["end"])
            if cur is None or e.get("filed", "") > cur[2]:
                best[e["end"]] = (float(e["val"]), concept, e.get("filed", ""), e.get("form"))
        for end, v in best.items():
            out.setdefault(end, v)  # earlier candidates take priority; later ones only fill missing years
    return out


def normalize(facts, years=5):
    """companyfacts JSON -> (inc, cf, bs, meta) with the row names the rest of the app expects. Missing stays missing."""
    S = {k: annual_series(facts, v, False) for k, v in FLOW.items()}
    S.update({k: annual_series(facts, v, True) for k, v in INSTANT.items()})
    ends = sorted(S["Total Revenue"].keys(), reverse=True)[:years]
    meta = {"entity": facts.get("entityName"), "cik": facts.get("cik"), "missing": [], "provenance": {}, "years": len(ends)}
    if not ends:
        meta["missing"].append("Revenue (no annual 10-K revenue facts found)")
        return None, None, None, meta
    g = lambda k, e: S[k][e][0] if e in S[k] else None
    inc, cf, bs = {}, {}, {}
    for e in ends:
        rev, cost, gp = g("Total Revenue", e), g("Cost Of Revenue", e), g("Gross Profit", e)
        if gp is None and rev is not None and cost is not None:
            gp = rev - cost
        cfo, capex = g("Operating Cash Flow", e), g("Capex", e)
        fcf = cfo - abs(capex) if cfo is not None and capex is not None else None
        debt = g("LT Debt Total", e)
        if debt is None:
            parts = [g("LT Debt Noncurrent", e), g("Debt Current", e)]
            debt = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
        if debt is not None or g("ST Borrowings", e) is not None or g("Finance Lease Liability", e) is not None:
            debt = (debt or 0) + (g("ST Borrowings", e) or 0) + (g("Finance Lease Liability", e) or 0)
        cash = g("Cash And ST Investments", e)
        if cash is None and g("Cash", e) is not None:
            cash = g("Cash", e) + (g("ST Investments", e) or 0)
        inc[e] = {"Total Revenue": rev, "Operating Income": g("Operating Income", e), "Net Income": g("Net Income", e), "Gross Profit": gp,
                  "EBIT": g("Operating Income", e), "Interest Expense": g("Interest Expense", e)}
        cf[e] = {"Operating Cash Flow": cfo, "Free Cash Flow": fcf, "Stock Based Compensation": g("Stock Based Compensation", e),
                 "Capital Expenditure": -abs(capex) if capex is not None else None, "Depreciation And Amortization": g("Depreciation And Amortization", e)}
        bs[e] = {"Total Assets": g("Total Assets", e), "Stockholders Equity": g("Stockholders Equity", e), "Goodwill": g("Goodwill", e),
                 "Accounts Receivable": g("Accounts Receivable", e), "Inventory": g("Inventory", e), "Minority Interest": g("Minority Interest", e),
                 "Total Debt": debt, "Cash Cash Equivalents And Short Term Investments": cash, "Ordinary Shares Number": g("Shares Outstanding (FY end)", e)}
    frames = []
    for d in (inc, cf, bs):
        df = pd.DataFrame(d).astype(float)
        frames.append(df.dropna(how="all"))
    inc, cf, bs = frames
    latest = ends[0]
    meta["latest_fy_end"] = latest
    meta["age_days"] = (date.today() - datetime.strptime(latest, "%Y-%m-%d").date()).days
    meta["filed"] = S["Total Revenue"][latest][2]
    meta["provenance"] = {k: {"concept": S[k][latest][1], "filed": S[k][latest][2]} for k in S if latest in S[k]}
    meta["missing"] = [k for k in ("Operating Income", "Net Income", "Operating Cash Flow", "Total Debt", "Total Assets") if
                       not any(k in df.index for df in (inc, cf, bs))]
    dei = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    if dei:
        last = max(dei, key=lambda x: x.get("filed", ""))
        meta["shares_cover"], meta["shares_cover_date"] = float(last["val"]), last.get("end")
    dil, bas = g("Diluted Average Shares", latest), g("Basic Average Shares", latest)
    meta["dilution_factor"] = dil / bas if dil and bas and 1.0 <= dil / bas <= 1.15 else 1.0
    return inc, cf, bs, meta


def fallback_info(meta, inc, cf, bs, price):
    """Fields the valuation needs when Yahoo's info table is thin. Every value comes straight from SEC filings."""
    info, note = {}, {}
    if inc is None:
        return info, note
    L = lambda df, r: float(df.loc[r].dropna().iloc[0]) if df is not None and r in df.index and df.loc[r].notna().any() else None
    rev = L(inc, "Total Revenue"); oi = L(inc, "Operating Income"); ni = L(inc, "Net Income"); gp = L(inc, "Gross Profit")
    if rev:
        info["totalRevenue"] = rev; note["totalRevenue"] = f"SEC 10-K FY ending {meta['latest_fy_end']} (annual, not TTM)"
        if oi is not None: info["operatingMargins"] = oi / rev
        if ni is not None: info["profitMargins"] = ni / rev
        if gp is not None: info["grossMargins"] = gp / rev
    debt, cash = L(bs, "Total Debt"), L(bs, "Cash Cash Equivalents And Short Term Investments")
    if debt is not None: info["totalDebt"] = debt
    if cash is not None: info["totalCash"] = cash
    fcf = L(cf, "Free Cash Flow")
    if fcf is not None: info["freeCashflow"] = fcf
    sh = meta.get("shares_cover")
    if sh:
        sh_dil = sh * meta.get("dilution_factor", 1.0)
        info["sharesOutstanding"] = sh_dil
        note["sharesOutstanding"] = f"SEC cover page {meta.get('shares_cover_date')}, x{meta.get('dilution_factor', 1.0):.3f} dilution factor from diluted/basic weighted shares"
        if price: info["marketCap"] = sh_dil * price; note["marketCap"] = "SEC shares x price"
    return info, note
