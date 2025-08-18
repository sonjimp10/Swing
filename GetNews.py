
# NewsFeed.py — multi-source (Bing RSS + Yahoo News RSS + Finviz) with OG-title/body and timestamps.
# No command-line args. Put 'tickers.csv' (column 'Ticker') next to this file and press Run.

import csv, html, random, re, sys, time, urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from zoneinfo import ZoneInfo  # py>=3.9

# =========================
# CONFIG
# =========================
TICKERS_CSV   = "tickers.csv"        # input CSV with column 'Ticker'
OUTPUT_CSV    = "news_clean.csv"     # output CSV
PER_TICKER    = 16                   # target items per ticker (combined sources)
SINCE_DAYS    = 14                   # ignore items older than this if timestamp is present
SLEEP_SECONDS = (0.25, 0.6)          # polite random delay
TIMEOUT_SEC   = 20

# Trusted finance + PR + regulator domains
GOOD_DOMAINS = {
    # finance/news
    "reuters.com","bloomberg.com","wsj.com","cnbc.com","finance.yahoo.com",
    "marketwatch.com","ft.com","barrons.com","benzinga.com","seekingalpha.com",
    "investing.com","thestreet.com","fool.com","investorplace.com","ibd.com",
    "forbes.com","marketbeat.com","morningstar.com","zacks.com","tipranks.com",
    # PR / filings / regulator (lets you catch FDA approvals / company releases)
    "businesswire.com","globenewswire.com","prnewswire.com","newsfilecorp.com",
    "fda.gov","novonordisk.com","investor.novonordisk.com"
}

# Optional publisher filters (case-insensitive substrings)
PUBLISHER_ALLOW: List[str] = []      # e.g., ["Reuters","Bloomberg"]
PUBLISHER_BLOCK: List[str] = []      # e.g., ["MSN"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BASE_HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

# =========================
# helpers
# =========================
def jitter_sleep():
    lo, hi = SLEEP_SECONDS if isinstance(SLEEP_SECONDS, tuple) else (SLEEP_SECONDS, SLEEP_SECONDS + 0.1)
    time.sleep(random.uniform(lo, hi))

def ensure_tickers_csv(path: Path):
    if path.exists(): return
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("Ticker\nAAPL\nMSFT\nNVDA\nNVO\nPANW\nMETA\nENPH\n")
    print(f"[INFO] Created sample {path}")

def read_tickers(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError(f"{path} must have a 'Ticker' column. Found: {df.columns.tolist()}")
    return (
        df["Ticker"].dropna().astype(str).str.upper().str.strip()
        .loc[lambda s: s != ""].drop_duplicates().tolist()
    )

def to_iso_from_struct(t) -> Optional[str]:
    try:
        return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
    except Exception:
        return None

def domain_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc.split(":")[0].lower()
    except Exception:
        return ""

def passes_publisher_filters(publisher: str) -> bool:
    if PUBLISHER_ALLOW and not any(a.lower() in publisher.lower() for a in PUBLISHER_ALLOW):
        return False
    if PUBLISHER_BLOCK and any(b.lower() in publisher.lower() for b in PUBLISHER_BLOCK):
        return False
    return True

# ---- read OG/meta from publisher page, including publish time if present ----
PUBLISHED_META_KEYS = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "article:published_time"}),
    ("meta", {"name": "pubdate"}),
    ("time", {"datetime": True})
]

def get_meta_title_desc_time(session: requests.Session, url: str) -> Tuple[str, str, Optional[str]]:
    try:
        r = session.get(url, timeout=TIMEOUT_SEC, allow_redirects=True)
        r.raise_for_status()
    except Exception:
        return "", "", None

    soup = BeautifulSoup(r.text, "html.parser")

    # title
    titles=[]
    t = soup.find("title")
    if t and t.text: titles.append(t.text.strip())
    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"): titles.append(ogt["content"].strip())
    best_title = max(titles, key=len) if titles else ""

    # description
    descs=[]
    ogd = soup.find("meta", property="og:description")
    if ogd and ogd.get("content"): descs.append(ogd["content"].strip())
    md = soup.find("meta", attrs={"name":"description"})
    if md and md.get("content"): descs.append(md["content"].strip())
    best_desc = max(descs, key=len) if descs else ""

    # published time
    ts_iso = None
    for tag, attrs in PUBLISHED_META_KEYS:
        node = soup.find(tag, attrs)
        if not node: continue
        val = node.get("content") if tag == "meta" else node.get("datetime")
        if not val: continue
        # normalize to ISO with timezone where possible
        try:
            # many sites already provide RFC3339/ISO
            ts_iso = datetime.fromisoformat(val.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            break
        except Exception:
            # try common formats
            for fmt in ("%Y-%m-%d %H:%M:%S%z", "%a, %d %b %Y %H:%M:%S %Z"):
                try:
                    ts_iso = datetime.strptime(val, fmt).astimezone(timezone.utc).isoformat()
                    break
                except Exception:
                    pass
            if ts_iso: break

    return html.unescape(best_title), html.unescape(best_desc), ts_iso

# =========================
# Source 1: Bing News RSS (broad query; filter by domain here)
# =========================
def fetch_bing(session: requests.Session, ticker: str, per_ticker: int) -> List[Dict]:
    try:
        import feedparser
    except ImportError:
        print("[WARN] install feedparser: pip install feedparser")
        return []

    q = urllib.parse.quote(
        f'"{ticker}" (stock OR shares OR earnings OR guidance OR results OR FDA OR approval OR upgrade OR downgrade)'
    )
    url = f"https://www.bing.com/news/search?q={q}&format=rss&setlang=en"
    feed = feedparser.parse(url)

    out=[]
    for e in feed.entries[: per_ticker * 3]:
        link   = getattr(e, "link", None)
        title  = getattr(e, "title", "") or ""
        pub_ts = to_iso_from_struct(getattr(e, "published_parsed", None)) if getattr(e, "published_parsed", None) else None
        src    = getattr(e, "source", {})
        publisher = src.get("title") if isinstance(src, dict) else (getattr(e, "source_title", "") or "")

        if not link: continue
        dom = domain_of(link)
        if not any(dom.endswith(gd) for gd in GOOD_DOMAINS):  # strict filter here
            continue
        if not passes_publisher_filters(publisher or dom):
            continue

        out.append({
            "source": "bing_rss",
            "publisher": publisher or dom,
            "link": link,
            "title_hint": title,
            "published_utc": pub_ts
        })
        if len(out) >= per_ticker:
            break
    return out

# =========================
# Source 2: Yahoo News Search RSS
# =========================
def fetch_yahoo_news_rss(session: requests.Session, ticker: str, per_ticker: int) -> List[Dict]:
    try:
        import feedparser
    except ImportError:
        print("[WARN] install feedparser: pip install feedparser")
        return []
    q = urllib.parse.quote(
        f'"{ticker}" (stock OR shares OR earnings OR guidance OR results OR FDA OR approval OR upgrade OR downgrade)'
    )
    url = f"https://news.search.yahoo.com/rss?p={q}"
    feed = feedparser.parse(url)

    out=[]
    for e in feed.entries[: per_ticker * 3]:
        link   = getattr(e, "link", None)
        title  = getattr(e, "title", "") or ""
        pub_ts = to_iso_from_struct(getattr(e, "published_parsed", None)) if getattr(e, "published_parsed", None) else None
        publisher = getattr(e, "source", {}).get("title") if hasattr(e, "source") else (getattr(e, "author", "") or "")

        if not link: continue
        dom = domain_of(link)
        if not any(dom.endswith(gd) for gd in GOOD_DOMAINS):
            continue
        if not passes_publisher_filters(publisher or dom):
            continue

        out.append({
            "source": "yahoo_rss",
            "publisher": publisher or dom,
            "link": link,
            "title_hint": title,
            "published_utc": pub_ts
        })
        if len(out) >= per_ticker:
            break
    return out

# =========================
# Source 3: Finviz (per-ticker table) with timestamp parsing
# =========================
FINVIZ_DT_PATTERNS = [
    ("%b-%d-%y %I:%M%p", True),   # "Aug-15-25 09:42AM"
    ("%b-%d-%y", False),          # "Aug-15-25" (Finviz sometimes omits time on repeated date rows)
]

def parse_finviz_dt(s: str) -> Optional[str]:
    s = s.strip()
    for fmt, has_time in FINVIZ_DT_PATTERNS:
        try:
            dt_naive = datetime.strptime(s, fmt)
            # Finviz times are in US/Eastern
            if has_time:
                dt_local = dt_naive.replace(tzinfo=ZoneInfo("US/Eastern"))
            else:
                # assume midnight Eastern if time omitted
                dt_local = dt_naive.replace(hour=0, minute=0, tzinfo=ZoneInfo("US/Eastern"))
            return dt_local.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return None

def fetch_finviz(session: requests.Session, ticker: str, per_ticker: int) -> List[Dict]:
    url = f"https://finviz.com/quote.ashx?t={urllib.parse.quote(ticker)}"
    try:
        r = session.get(url, timeout=TIMEOUT_SEC)
        r.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", class_="fullview-news-outer")
    if not table: return []

    out=[]; current_date_prefix=None
    for row in table.select("tr"):
        tds = row.find_all("td")
        if len(tds) < 2: continue
        # Finviz uses the first td for "Date" or "Time"
        dt_text = tds[0].get_text(strip=True)
        # Sometimes the first cell is only time; use last seen date prefix
        if re.match(r"^\d{1,2}:\d{2}[AP]M$", dt_text):
            if current_date_prefix:
                finviz_ts = parse_finviz_dt(f"{current_date_prefix} {dt_text}")
            else:
                finviz_ts = None
        else:
            # e.g., "Aug-15-25 09:42AM" or "Aug-15-25"
            finviz_ts = parse_finviz_dt(dt_text)
            # remember date part if present
            m = re.match(r"^([A-Za-z]{3}-\d{2}-\d{2})", dt_text)
            if m: current_date_prefix = m.group(1)

        a = tds[1].find("a", href=True)
        if not a: continue
        link = a["href"]
        title = a.get_text(strip=True)
        dom = domain_of(link)
        if not any(dom.endswith(gd) for gd in GOOD_DOMAINS):
            continue

        out.append({
            "source": "finviz",
            "publisher": dom,
            "link": link,
            "title_hint": title,
            "published_utc": finviz_ts
        })
        if len(out) >= per_ticker: break
    return out

# =========================
# MAIN
# =========================
def main():
    base = Path(__file__).resolve().parent
    tickers_path = base / TICKERS_CSV
    ensure_tickers_csv(tickers_path)

    try:
        tickers = read_tickers(tickers_path)
    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)
    if not tickers:
        print("[ERROR] No tickers in tickers.csv"); sys.exit(1)

    since_cutoff = None
    if SINCE_DAYS and SINCE_DAYS > 0:
        since_cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=SINCE_DAYS)

    bing_session  = requests.Session();  bing_session.headers.update(BASE_HEADERS)
    yahoo_session = requests.Session(); yahoo_session.headers.update(BASE_HEADERS)
    fin_session   = requests.Session();   fin_session.headers.update(BASE_HEADERS)
    page_session  = requests.Session();  page_session.headers.update(BASE_HEADERS)

    rows=[]; seen=set()

    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...")

        items = []
        items += fetch_bing(bing_session, t, PER_TICKER)
        if len(items) < PER_TICKER:
            items += fetch_yahoo_news_rss(yahoo_session, t, PER_TICKER - len(items))
        if len(items) < PER_TICKER:
            items += fetch_finviz(fin_session, t, PER_TICKER - len(items))

        # Enrich & save
        for it in items:
            url = it["link"]
            if url in seen: 
                continue

            ts = it.get("published_utc")

            # If still missing a date, try to read it from the article page
            title, desc, page_ts = get_meta_title_desc_time(page_session, url)
            if not ts and page_ts:
                ts = page_ts

            # filter by age if we now have ts
            if since_cutoff and ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                    if dt < since_cutoff:
                        continue
                except Exception:
                    pass

            final_title = (title or it.get("title_hint") or "").strip()
            publisher = (it.get("publisher") or domain_of(url)).strip()
            if not passes_publisher_filters(publisher):
                continue

            rows.append({
                "ticker": t,
                "source": it["source"],
                "publisher": publisher,
                "domain": domain_of(url),
                "title": final_title,
                "published_utc": ts,
                "final_url": url,
                "snippet": " ".join((desc or "").split()[:60]),
                "ImpactScore": 0  # scoring later; first, nail clean extraction
            })
            seen.add(url)
            jitter_sleep()

    # Sort newest (if timestamps present)
    rows.sort(key=lambda r: (r.get("published_utc") or "", r["publisher"], r["title"]), reverse=True)

    out_path = base / OUTPUT_CSV
    cols = ["ticker","source","publisher","domain","title","published_utc","final_url","snippet","ImpactScore"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"\n Saved {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
