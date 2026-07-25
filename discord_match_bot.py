import os
import json
import re
import ast
import sys
import time
import traceback
import asyncio
import signal
import subprocess
import shutil
import urllib.parse
import urllib.request
import discord
from discord.ext import commands
from openai import AsyncOpenAI, APIStatusError

# ── env config ──────────────────────────────────────────────
DEFAULT_MIN_EDGE = float(os.getenv("MIN_EDGE", "0.05"))
SURPLUS_API_KEY  = os.getenv("SURPLUS_API_KEY")
SURPLUS_BASE_URL = os.getenv("SURPLUS_API_URL", "https://api.surplusintelligence.ai/min30/v1")
SURPLUS_MODEL    = os.getenv("SURPLUS_MODEL", "gpt-5.4")
CONFIG_FILE      = "watcher_config.json"
MONITOR_CHANNEL_ID = 1530286757126471822

# discord limits
EMBED_FIELD_MAX = 1024
EMBED_TOTAL_MAX = 5500

def _trunc(s: str, max_len: int = EMBED_FIELD_MAX) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."

# ── discord setup ───────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# per-guild min_edge
user_settings = {}

def get_min_edge(guild_id: int) -> float:
    return user_settings.get(guild_id, {}).get("min_edge", DEFAULT_MIN_EDGE)

def set_min_edge(guild_id: int, edge: float):
    if guild_id not in user_settings:
        user_settings[guild_id] = {}
    user_settings[guild_id]["min_edge"] = edge

# ── polymarket url parsing ──────────────────────────────────
# Matches BOTH URL formats and extracts slug + optional sport category:
#   https://polymarket.com/event/kor-poh-jeo-2026-07-25
#   https://polymarket.com/sports/mma/ufc-fight-night-2026-07-25
#   https://polymarket.com/sports/swe/swe-deg-dju-2026-07-25
_POLYMARKET_URL_RE = re.compile(
    r'https?://polymarket\.com/(?:event|sports/([a-z0-9]+))/([a-z0-9][a-z0-9\-]+[a-z0-9])',
    re.IGNORECASE
)

def extract_slug_and_category_from_url(url: str) -> tuple[str, str]:
    """Extract the event slug and optional sport category from a Polymarket URL.
    Returns (slug, category) where category may be None for /event/ URLs."""
    m = _POLYMARKET_URL_RE.search(url)
    if m:
        category = m.group(1).lower() if m.group(1) else None
        slug = m.group(2)
        return slug, category
    return None, None

# For backward compatibility
def extract_slug_from_url(url: str) -> str:
    slug, _ = extract_slug_and_category_from_url(url)
    return slug

def fetch_event_from_gamma(slug: str) -> dict:
    """Query Polymarket Gamma API for event details by slug.
    Returns the event dict or None."""
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
    except Exception as e:
        print(f"[Gamma API] Event lookup failed for slug '{slug}': {e}", file=sys.stderr)
    return None

# ── sport detection from Gamma API event ────────────────────
_GAMMA_SPORT_MAP = {
    # sport.sport field
    "mlb": "mlb", "nba": "nba", "nfl": "nfl", "nhl": "nhl",
    "epl": "soccer", "laliga": "soccer", "seriea": "soccer",
    "bundesliga": "soccer", "ligue1": "soccer", "mls": "soccer",
    "ucl": "soccer", "uel": "soccer", "kor": "soccer",
    "j1": "soccer", "eredivisie": "soccer", "liga": "soccer",
    "bra": "soccer", "arg": "soccer", "afl": "afl",
    "ncaa": "ncaa", "ncaaf": "ncaa", "ncaab": "ncaa",
    "wnba": "wnba", "cricket": "cricket", "tennis": "tennis",
    "mma": "mma", "boxing": "boxing", "f1": "f1",
    "swe": "soccer",
    # tag labels (lowercase)
    "soccer": "soccer", "baseball": "mlb", "basketball": "nba",
    "football": "nfl", "hockey": "nhl", "k-league": "soccer",
    "premier league": "soccer", "la liga": "soccer",
    "serie a": "soccer", "bundesliga": "soccer",
    "ligue 1": "soccer", "mls": "soccer",
    "champions league": "soccer", "europa league": "soccer",
    "tennis": "tennis", "cricket": "cricket",
    "mma": "mma", "ufc": "mma",           # ← UFC aliased to MMA
    "boxing": "boxing", "formula 1": "f1",
    "afl": "afl", "ncaa": "ncaa", "wnba": "wnba",
    "allsvenskan": "soccer",
}

# URL path category → our internal sport (for /sports/<cat>/ URLs)
_URL_CATEGORY_SPORT_MAP = {
    "mma": "mma", "ufc": "mma",
    "boxing": "boxing",
    "tennis": "tennis",
    "f1": "f1", "formula1": "f1",
    "mlb": "mlb", "baseball": "mlb",
    "nba": "nba", "basketball": "nba",
    "nfl": "nfl", "football": "nfl",
    "nhl": "nhl", "hockey": "nhl",
    "soccer": "soccer", "swe": "soccer", "epl": "soccer",
    "kor": "soccer", "j1": "soccer", "bra": "soccer", "arg": "soccer",
    "cricket": "cricket",
    "afl": "afl",
    "ncaa": "ncaa",
    "wnba": "wnba",
}

# ── INDIVIDUAL SPORTS ───────────────────────────────────────
INDIVIDUAL_SPORTS = {"tennis", "mma", "boxing", "f1"}

def sport_has_draws(sport: str) -> bool:
    no_draw = {"mlb", "nba", "nfl", "nhl", "tennis", "mma", "boxing", "f1"}
    return sport not in no_draw

def is_individual_sport(sport: str) -> bool:
    return sport in INDIVIDUAL_SPORTS

def detect_sport_from_url_category(category: str) -> str:
    """Detect sport from the /sports/<category>/ URL path segment.
    This is a high-confidence signal — the URL itself tells us the sport."""
    if not category:
        return None
    cat = category.lower().strip()
    return _URL_CATEGORY_SPORT_MAP.get(cat)

def detect_sport_from_gamma_event(event: dict) -> str:
    """Extract the sport from a Gamma API event object."""
    if not event or not isinstance(event, dict):
        return "unknown"

    sport_obj = event.get("sport")
    if isinstance(sport_obj, dict):
        sport_code = sport_obj.get("sport", "").lower().strip()
        if sport_code and sport_code in _GAMMA_SPORT_MAP:
            return _GAMMA_SPORT_MAP[sport_code]

    series_slug = event.get("seriesSlug", "").lower().strip()
    if series_slug and series_slug in _GAMMA_SPORT_MAP:
        return _GAMMA_SPORT_MAP[series_slug]

    tags = event.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                label = tag.get("label", "").lower().strip()
                if label and label in _GAMMA_SPORT_MAP:
                    return _GAMMA_SPORT_MAP[label]
                slug = tag.get("slug", "").lower().strip()
                if slug and slug in _GAMMA_SPORT_MAP:
                    return _GAMMA_SPORT_MAP[slug]

    return "unknown"

# ── bullpen CLI wrapper & watcher state ─────────────────────
def load_watcher_config() -> dict:
    defaults = {
        "active": False,
        "channel_id": MONITOR_CHANNEL_ID,
        "address": None,
        "source": None,
        "stop_loss": None,
        "auto_close_sl": True,
        "interval": 2,
        "periodic_interval": 300,
        "sl_alert_fired": False,
        "last_positions": None
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
                defaults.update(saved)
                if not saved.get("channel_id"):
                    defaults["channel_id"] = MONITOR_CHANNEL_ID
        except Exception as e:
            print(f"[Config Load Error] {e}", file=sys.stderr)
    return defaults

def save_watcher_config(state: dict):
    try:
        to_save = {
            "active": state.get("active", False),
            "channel_id": state.get("channel_id", MONITOR_CHANNEL_ID),
            "address": state.get("address"),
            "source": state.get("source"),
            "stop_loss": state.get("stop_loss"),
            "auto_close_sl": state.get("auto_close_sl", True),
            "interval": state.get("interval", 2),
            "periodic_interval": state.get("periodic_interval", 300),
            "sl_alert_fired": state.get("sl_alert_fired", False),
            "last_positions": state.get("last_positions")
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(to_save, f, indent=2)
    except Exception as e:
        print(f"[Config Save Error] {e}", file=sys.stderr)

bullpen_watch_state = load_watcher_config()
bullpen_watch_state["task"] = None
last_periodic_time = 0

def get_bullpen_binary_path() -> str:
    priority_paths = [
        "/root/.bullpen/bin/bullpen",
        os.path.expanduser("~/.bullpen/bin/bullpen"),
    ]
    for p in priority_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    candidate = shutil.which("bullpen")
    if candidate and candidate != "/usr/local/bin/bullpen":
        return candidate
    fallback_paths = ["/usr/local/bin/bullpen", "/usr/bin/bullpen"]
    for p in fallback_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return "bullpen"

def run_bullpen_positions(address: str = None, source: str = None, json_mode: bool = False) -> str:
    bin_path = get_bullpen_binary_path()
    cmd_parts = [bin_path, "polymarket", "positions"]
    if json_mode:
        cmd_parts.append("--json")
    if address:
        cmd_parts.extend(["--address", address])
    if source:
        cmd_parts.extend(["--source", source])
    env = os.environ.copy()
    bullpen_bin_dir = os.path.expanduser("~/.bullpen/bin")
    if os.path.exists(bullpen_bin_dir):
        env["PATH"] = f"{bullpen_bin_dir}:{env.get('PATH', '')}"
    try:
        res = subprocess.run(cmd_parts, capture_output=True, text=True, check=True, timeout=10, env=env)
        return res.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Error] `bullpen` CLI command timed out after 10 seconds."
    except FileNotFoundError:
        return "[Error] `bullpen` CLI tool is not installed or not in PATH on this server."
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() or e.stdout.strip()
        return f"[Bullpen Error] {err_msg}"
    except Exception as e:
        return f"[Error] Failed to execute bullpen CLI: {e}"

# ── TICKER EXTRACTION ───────────────────────────────────────
_TICKER_RE = re.compile(
    r'\b([a-z]{2,15}-[a-z]{2,15}-[a-z]{2,15}-\d{4}-\d{2}-\d{2})\b',
    re.IGNORECASE
)
_PAREN_TICKER_RE = re.compile(
    r'\(([a-z][a-z0-9\-]{5,}[a-z0-9])\)',
    re.IGNORECASE
)

def _extract_all_tickers(text: str, exclude_slug: str = None) -> list[str]:
    candidates = []
    for m in _TICKER_RE.finditer(text):
        ticker = m.group(1).lower()
        if ticker not in candidates:
            candidates.append(ticker)
    for m in _PAREN_TICKER_RE.finditer(text):
        slug = m.group(1).lower()
        if '-' in slug and slug not in candidates:
            candidates.append(slug)
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r'^\d+\.', stripped):
            parens = re.findall(r'\(([^)]+)\)', stripped)
            if parens:
                last = parens[-1].strip().lower()
                if '-' in last and len(last) > 5 and last not in candidates:
                    candidates.append(last)
    if exclude_slug:
        candidates = [c for c in candidates if c != exclude_slug.lower()]
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

def execute_bullpen_sell(market_slug: str, outcome: str, shares: float) -> str:
    bin_path = get_bullpen_binary_path()
    cmd_parts = [bin_path, "polymarket", "sell", market_slug, outcome, str(shares)]
    env = os.environ.copy()
    bullpen_bin_dir = os.path.expanduser("~/.bullpen/bin")
    if os.path.exists(bullpen_bin_dir):
        env["PATH"] = f"{bullpen_bin_dir}:{env.get('PATH', '')}"
    try:
        res = subprocess.run(cmd_parts, capture_output=True, text=True, check=True, timeout=15, env=env)
        return res.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Error] `bullpen polymarket sell` command timed out."
    except subprocess.CalledProcessError as e:
        combined = (e.stdout or "") + "\n" + (e.stderr or "")
        if not combined.strip():
            combined = getattr(e, 'output', '') or ''
        print(f"[Sell Debug] combined output ({len(combined)} chars): {combined[:500]}", file=sys.stderr)
        all_candidates = _extract_all_tickers(combined, exclude_slug=market_slug)
        print(f"[Sell Debug] Extracted candidates: {all_candidates}", file=sys.stderr)
        if all_candidates:
            for ticker in all_candidates[:5]:
                print(f"[Sell Instant Retry] Retrying with suggested ticker: {ticker}", file=sys.stderr)
                retry_parts = [bin_path, "polymarket", "sell", ticker, outcome, str(shares)]
                try:
                    res_retry = subprocess.run(retry_parts, capture_output=True, text=True, check=True, timeout=15, env=env)
                    print(f"[Sell Instant Retry] SUCCESS with ticker: {ticker}", file=sys.stderr)
                    return res_retry.stdout.strip()
                except Exception as retry_err:
                    combined_retry = (getattr(retry_err, 'stdout', '') or '') + "\n" + (getattr(retry_err, 'stderr', '') or '')
                    print(f"[Sell Retry Failed for {ticker}] {combined_retry[:200]}", file=sys.stderr)
        err_out = combined.strip() or f"Exit code {e.returncode}"
        return f"[Sell Error] Exit code {e.returncode}: {_trunc(err_out, 900)}"
    except Exception as e:
        return f"[Sell Error] Failed to sell position: {e}"

# ── SLUG RESOLUTION ─────────────────────────────────────────
_slug_cache = {}
_json_keys_logged = False
_SLUG_KEY_PATTERNS = re.compile(r'slug|ticker|event', re.IGNORECASE)
_SPORTS_REQUIRING_TICKERS = {"mlb", "nba", "nfl", "nhl"}

def _detect_sport_from_title(title: str) -> str:
    if not title:
        return "unknown"
    t = title.lower()
    for sport in _SPORTS_REQUIRING_TICKERS:
        if sport in t:
            return sport
    for team, sport in TEAM_SPORT_MAP.items():
        if team in t:
            return sport
    return "unknown"

def _is_truncated(s: str) -> bool:
    if not s:
        return True
    return s.rstrip().endswith("...") or s.rstrip().endswith("…")

def _extract_slug_from_bullpen_json(item: dict) -> str:
    if not isinstance(item, dict):
        return None
    for key in ["slug", "marketSlug", "eventSlug", "eventTicker", "ticker", "market_slug", "event_slug", "event_ticker"]:
        val = item.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 3 and not _is_truncated(val):
            return val.strip()
    for key, val in item.items():
        if not isinstance(val, str) or len(val.strip()) <= 3:
            continue
        if _is_truncated(val):
            continue
        if _SLUG_KEY_PATTERNS.search(key):
            if re.match(r'^[a-z0-9][a-z0-9\-]{3,}[a-z0-9]$', val.strip()):
                return val.strip()
    return None

def _extract_token_ids_from_bullpen_json(item: dict) -> list[str]:
    ids = []
    for key in ["conditionId", "condition_id", "conditionID",
                "tokenId", "token_id", "tokenID",
                "clobTokenId", "clob_token_id",
                "marketId", "market_id", "marketID",
                "assetId", "asset_id", "assetID",
                "id", "_id"]:
        val = item.get(key)
        if val:
            ids.append(str(val))
    for key in ["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"]:
        val = item.get(key)
        if isinstance(val, list):
            for v in val:
                if v:
                    ids.append(str(v))
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    for v in parsed:
                        if v:
                            ids.append(str(v))
            except (json.JSONDecodeError, TypeError):
                if val:
                    ids.append(val)
    return ids

def fetch_exact_slug_by_id(token_or_condition_id: str) -> str:
    if not token_or_condition_id:
        return None
    if token_or_condition_id in _slug_cache:
        return _slug_cache[token_or_condition_id]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }
    endpoints = [
        f"https://gamma-api.polymarket.com/markets?clob_token_ids={token_or_condition_id}",
        f"https://gamma-api.polymarket.com/markets?condition_id={token_or_condition_id}",
        f"https://gamma-api.polymarket.com/markets/{token_or_condition_id}",
    ]
    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    items = data if isinstance(data, list) else [data]
                    for m in items:
                        if isinstance(m, dict):
                            slug = m.get("slug") or m.get("marketSlug") or m.get("market_slug")
                            if slug and isinstance(slug, str) and len(slug.strip()) > 3:
                                clean_slug = re.sub(r'-+', '-', slug.strip())
                                _slug_cache[token_or_condition_id] = clean_slug
                                print(f"[Gamma API] Resolved slug: {clean_slug} from ID: {token_or_condition_id}", file=sys.stderr)
                                return clean_slug
        except Exception as e:
            print(f"[Gamma API] Endpoint failed ({url}): {e}", file=sys.stderr)
            continue
    _slug_cache[token_or_condition_id] = None
    print(f"[Gamma API] No slug found for ID: {token_or_condition_id}", file=sys.stderr)
    return None

def resolve_exact_slug(market_title: str, item_dict: dict = None) -> str:
    global _json_keys_logged
    sport = _detect_sport_from_title(market_title)
    if item_dict and isinstance(item_dict, dict):
        if not _json_keys_logged:
            _json_keys_logged = True
            print(f"[DEBUG] Bullpen JSON keys: {list(item_dict.keys())}", file=sys.stderr)
        slug = _extract_slug_from_bullpen_json(item_dict)
        if slug:
            print(f"[Slug] Found in bullpen JSON: {slug}", file=sys.stderr)
            return slug
        ids = _extract_token_ids_from_bullpen_json(item_dict)
        for id_val in ids:
            slug = fetch_exact_slug_by_id(id_val)
            if slug:
                return slug
    if sport in _SPORTS_REQUIRING_TICKERS:
        if market_title:
            clean = market_title.replace("…", "").replace("...", "").strip()
            search_slug = re.sub(r'[^a-z0-9\s]+', '', clean.lower())
            search_slug = re.sub(r'\s+', '-', search_slug.strip())
            print(f"[Slug] Sports market, using search slug (will trigger Did-you-mean retry): {search_slug}", file=sys.stderr)
            return search_slug
        return "sports-market"
    if market_title:
        clean = market_title.replace("…", "").replace("...", "").strip()
        s = clean.lower()
        s = re.sub(r'[^a-z0-9\s]+', '', s)
        s = re.sub(r'\s+', '-', s.strip())
        s = re.sub(r'-+', '-', s)
        return s.strip('-')
    return "unknown-market"

def parse_positions_to_cards(raw_output: str) -> tuple[str, list[dict]]:
    global _json_keys_logged
    if not raw_output or raw_output.startswith("[Error]") or raw_output.startswith("[Bullpen Error]"):
        return f"```\n{raw_output}\n```", []
    try:
        data = json.loads(raw_output)
        if isinstance(data, list):
            parsed_positions = []
            total_pnl = 0.0
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                if idx == 0 and not _json_keys_logged:
                    _json_keys_logged = True
                    print(f"[DEBUG] Bullpen JSON keys (item 0): {list(item.keys())}", file=sys.stderr)
                    try:
                        item_str = json.dumps(item, default=str)
                        print(f"[DEBUG] Bullpen JSON item 0 (first 500 chars): {item_str[:500]}", file=sys.stderr)
                    except Exception:
                        pass
                title = item.get("title") or item.get("market") or item.get("question") or item.get("name") or "Unknown Market"
                outcome = item.get("outcome") or item.get("outcomeName") or item.get("side") or "Yes"
                m_slug = resolve_exact_slug(title, item)
                status = item.get("status", "open")
                shares = str(item.get("shares") or item.get("size") or item.get("amount") or "0")
                entry_val = item.get("avgPrice") or item.get("entry") or item.get("averagePrice") or 0.0
                entry = f"${entry_val:.2f}" if isinstance(entry_val, (int, float)) else str(entry_val)
                now_val = item.get("curPrice") or item.get("now") or item.get("currentPrice") or 0.0
                now = f"${now_val:.2f}" if isinstance(now_val, (int, float)) else str(now_val)
                val_amt = item.get("currentValue") or item.get("value") or item.get("positionValue") or 0.0
                val = f"${val_amt:.2f}" if isinstance(val_amt, (int, float)) else str(val_amt)
                pnl_num = item.get("pnl") or item.get("profitLoss") or item.get("unrealizedPnl") or 0.0
                if isinstance(pnl_num, (int, float)):
                    pnl_str = f"-${abs(pnl_num):.2f}" if pnl_num < 0 else f"${pnl_num:.2f}"
                    total_pnl += pnl_num
                else:
                    pnl_str = str(pnl_num)
                roe_num = item.get("percentPnl") or item.get("roe") or item.get("pnlPercent") or item.get("returnPercent") or 0.0
                if isinstance(roe_num, (int, float)):
                    roe_str = f"{roe_num:.1f}%"
                else:
                    roe_str = str(roe_num)
                parsed_positions.append({
                    "raw": json.dumps(item),
                    "market": title,
                    "slug": m_slug,
                    "outcome": outcome,
                    "status": status,
                    "shares": shares,
                    "entry": entry,
                    "now": now,
                    "value": val,
                    "pnl": pnl_str,
                    "roe": roe_str
                })
            pnl_fmt = f"-${abs(total_pnl):.2f}" if total_pnl < 0 else f"${total_pnl:.2f}"
            header = f"📊 **Positions ({len(parsed_positions)} open, Total P&L: {pnl_fmt})**\n• _Source: polymarket_\n"
            return header, parsed_positions
    except (json.JSONDecodeError, TypeError):
        pass
    lines = raw_output.splitlines()
    summary_line = ""
    source_line = ""
    table_started = False
    position_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if l.startswith("Positions ("):
            summary_line = l
        elif l.startswith("Source:"):
            source_line = l
        elif l.startswith("---"):
            table_started = True
        elif table_started:
            position_lines.append(l)
    header = ""
    if summary_line:
        header += f"📊 **{summary_line}**\n"
    if source_line:
        header += f"• _{source_line}_\n"
    parsed_positions = []
    if position_lines:
        for pos in position_lines:
            tokens = pos.split()
            if len(tokens) >= 9:
                roe = tokens[-1]
                pnl = tokens[-2]
                val = tokens[-3]
                now = tokens[-4]
                entry = tokens[-5]
                shares = tokens[-6]
                status = tokens[-7]
                outcome = tokens[-8]
                market = " ".join(tokens[:-8])
                slug = resolve_exact_slug(market, None)
                parsed_positions.append({
                    "raw": pos,
                    "market": market,
                    "slug": slug,
                    "outcome": outcome,
                    "status": status,
                    "shares": shares,
                    "entry": entry,
                    "now": now,
                    "value": val,
                    "pnl": pnl,
                    "roe": roe
                })
    return header, parsed_positions

def format_card_from_obj(p: dict) -> str:
    pnl_emoji = "🔴" if "-" in p["pnl"] or "-" in p["roe"] else "🟢"
    return (
        f"📌 `{p['slug']}`\n"
        f"• Outcome: **{p['outcome']}** | Status: `{p['status']}`\n"
        f"• Shares: `{p['shares']}` | Value: `{p['value']}`\n"
        f"• Entry: `{p['entry']}` ➔ Now: `{p['now']}`\n"
        f"• P&L: {pnl_emoji} `{p['pnl']}` (`{p['roe']}`)"
    )

async def send_positions_embeds(target, title: str, raw_output: str, addr: str = None, src: str = None):
    header, pos_objs = parse_positions_to_cards(raw_output)
    if not pos_objs:
        embed = discord.Embed(
            title=title,
            description=header or f"```\n{_trunc(raw_output, 2000)}\n```",
            color=discord.Color.gold()
        )
        if addr: embed.add_field(name="Address", value=addr, inline=True)
        if src: embed.add_field(name="Source", value=src, inline=True)
        await target.send(embed=embed)
        return
    cards = [format_card_from_obj(p) for p in pos_objs]
    embeds_to_send = []
    current_embed = discord.Embed(title=title, description=header, color=discord.Color.gold())
    if addr: current_embed.add_field(name="Address", value=addr, inline=True)
    if src: current_embed.add_field(name="Source", value=src, inline=True)
    current_field_content = ""
    current_field_count = 0
    current_char_count = len(title) + len(header)
    for i, card in enumerate(cards, 1):
        card_text = f"{card}\n\n"
        if len(current_field_content) + len(card_text) > 1000:
            current_embed.add_field(
                name=f"Positions ({i - current_field_count} - {i - 1})",
                value=current_field_content.strip(),
                inline=False
            )
            current_char_count += len(current_field_content)
            current_field_content = ""
            current_field_count = 0
        if len(current_embed.fields) >= 15 or current_char_count > EMBED_TOTAL_MAX:
            embeds_to_send.append(current_embed)
            current_embed = discord.Embed(title=f"{title} (Cont.)", color=discord.Color.gold())
            current_char_count = len(title)
        current_field_content += card_text
        current_field_count += 1
    if current_field_content:
        total_cards = len(cards)
        start_idx = total_cards - current_field_count + 1
        current_embed.add_field(
            name=f"Positions ({start_idx} - {total_cards})" if total_cards > 1 else "Open Positions",
            value=current_field_content.strip(),
            inline=False
        )
    embeds_to_send.append(current_embed)
    for emb in embeds_to_send:
        await target.send(embed=emb)

async def fetch_positions_output(addr: str = None, src: str = None) -> str:
    res = await asyncio.to_thread(run_bullpen_positions, addr, src, True)
    if res.startswith("[Error]") or res.startswith("[Bullpen Error]") or not res:
        res = await asyncio.to_thread(run_bullpen_positions, addr, src, False)
    return res

async def close_position_task(channel, tp: dict):
    market_slug = tp['slug']
    close_embed = discord.Embed(
        title="⚡ AUTO-CLOSING POSITION",
        description=f"Executing immediate market sell for `{market_slug}`...",
        color=discord.Color.orange()
    )
    await channel.send(embed=close_embed)
    try:
        shares_num = float(tp['shares'])
        sell_res = await asyncio.to_thread(execute_bullpen_sell, market_slug, tp['outcome'], shares_num)
        is_err = "Error" in sell_res or "[Sell Error]" in sell_res
        res_embed = discord.Embed(
            title="✅ POSITION CLOSED" if not is_err else "❌ AUTO-CLOSE RESULT",
            description=f"Market: `{market_slug}`\n```\n{_trunc(sell_res, 1800)}\n```",
            color=discord.Color.green() if not is_err else discord.Color.red()
        )
        await channel.send(embed=res_embed)
    except Exception as se:
        err_embed = discord.Embed(
            title="❌ AUTO-CLOSE FAILED",
            description=f"Failed to sell `{market_slug}`: {se}",
            color=discord.Color.dark_red()
        )
        await channel.send(embed=err_embed)

async def bullpen_watcher_loop():
    global last_periodic_time
    while True:
        loop_start = time.time()
        try:
            if bullpen_watch_state["active"]:
                target_channel_id = bullpen_watch_state.get("channel_id") or MONITOR_CHANNEL_ID
                channel = bot.get_channel(target_channel_id)
                if channel:
                    addr = bullpen_watch_state.get("address")
                    src = bullpen_watch_state.get("source")
                    sl = bullpen_watch_state.get("stop_loss")
                    auto_close = bullpen_watch_state.get("auto_close_sl", True)
                    periodic_interval = bullpen_watch_state.get("periodic_interval", 300)
                    pos_output = await fetch_positions_output(addr, src)
                    if pos_output.startswith("[Error]") or pos_output.startswith("[Bullpen Error]"):
                        print(f"[Bullpen Watcher CLI Warning] {pos_output}", file=sys.stderr)
                    else:
                        now = time.time()
                        header, pos_objs = parse_positions_to_cards(pos_output)
                        triggered_positions = []
                        if sl is not None:
                            for p in pos_objs:
                                roe_str = p.get("roe", "")
                                if "-" in roe_str:
                                    try:
                                        loss_val = abs(float(roe_str.replace("%", "").replace("-", "")))
                                        if loss_val >= sl:
                                            triggered_positions.append(p)
                                    except ValueError:
                                        pass
                        if triggered_positions:
                            if not bullpen_watch_state.get("sl_alert_fired", False):
                                embed = discord.Embed(
                                    title=f"🚨 BULLPEN STOP LOSS TRIGGERED ({sl}%)",
                                    description=f"Found **{len(triggered_positions)}** position(s) exceeding Stop Loss threshold of `{sl}%`!",
                                    color=discord.Color.red()
                                )
                                for tp in triggered_positions:
                                    embed.add_field(
                                        name=f"⚠️ BREACHED: {tp['slug']}",
                                        value=f"Outcome: **{tp['outcome']}** | Shares: `{tp['shares']}`\nLoss: `{tp['pnl']}` (`{tp['roe']}`)",
                                        inline=False
                                    )
                                await channel.send(embed=embed)
                                if auto_close:
                                    close_tasks = [close_position_task(channel, tp) for tp in triggered_positions]
                                    await asyncio.gather(*close_tasks)
                                bullpen_watch_state["sl_alert_fired"] = True
                                save_watcher_config(bullpen_watch_state)
                        else:
                            if bullpen_watch_state.get("sl_alert_fired", False):
                                bullpen_watch_state["sl_alert_fired"] = False
                                save_watcher_config(bullpen_watch_state)
                        if last_periodic_time == 0 or (now - last_periodic_time) >= periodic_interval:
                            await send_positions_embeds(channel, "📊 Bullpen 5-Min Position Report", pos_output, addr, src)
                            last_periodic_time = now
                        if pos_output != bullpen_watch_state.get("last_positions"):
                            bullpen_watch_state["last_positions"] = pos_output
                            save_watcher_config(bullpen_watch_state)
        except Exception as e:
            print(f"[Bullpen Watcher Loop Error] {e}", file=sys.stderr)
        elapsed = time.time() - loop_start
        configured_interval = float(bullpen_watch_state.get("interval", 2))
        sleep_time = max(0.1, configured_interval - elapsed)
        await asyncio.sleep(sleep_time)

# ── sport detection (text-based fallback) ───────────────────
NO_DRAW_SPORTS = {"mlb", "nba", "nfl", "nhl", "tennis", "mma", "boxing", "f1"}

TEAM_SPORT_MAP = {
    "diamondbacks": "mlb", "braves": "mlb", "orioles": "mlb", "red sox": "mlb",
    "cubs": "mlb", "white sox": "mlb", "reds": "mlb", "guardians": "mlb",
    "rockies": "mlb", "tigers": "mlb", "astros": "mlb", "royals": "mlb",
    "angels": "mlb", "dodgers": "mlb", "marlins": "mlb", "brewers": "mlb",
    "twins": "mlb", "mets": "mlb", "yankees": "mlb", "athletics": "mlb",
    "phillies": "mlb", "pirates": "mlb", "padres": "mlb", "giants": "mlb",
    "mariners": "mlb", "cardinals": "mlb", "rays": "mlb", "rangers": "mlb",
    "blue jays": "mlb", "nationals": "mlb",
    "celtics": "nba", "nets": "nba", "knicks": "nba", "76ers": "nba",
    "raptors": "nba", "bulls": "nba", "cavaliers": "nba", "pistons": "nba",
    "pacers": "nba", "bucks": "nba", "hawks": "nba", "hornets": "nba",
    "heat": "nba", "magic": "nba", "wizards": "nba", "nuggets": "nba",
    "timberwolves": "nba", "thunder": "nba", "trail blazers": "nba",
    "jazz": "nba", "warriors": "nba", "clippers": "nba", "lakers": "nba",
    "suns": "nba", "kings": "nba", "mavericks": "nba", "rockets": "nba",
    "grizzlies": "nba", "pelicans": "nba", "spurs": "nba",
    "patriots": "nfl", "dolphins": "nfl", "jets": "nfl", "bills": "nfl",
    "ravens": "nfl", "steelers": "nfl", "browns": "nfl", "bengals": "nfl",
    "texans": "nfl", "colts": "nfl", "jaguars": "nfl", "titans": "nfl",
    "broncos": "nfl", "chiefs": "nfl", "raiders": "nfl", "chargers": "nfl",
    "cowboys": "nfl", "commanders": "nfl", "eagles": "nfl",
    "packers": "nfl", "lions": "nfl", "vikings": "nfl", "bears": "nfl",
    "buccaneers": "nfl", "saints": "nfl", "panthers": "nfl", "falcons": "nfl",
    "rams": "nfl", "49ers": "nfl", "seahawks": "nfl", "cardinals": "nfl",
    "bruins": "nhl", "sabres": "nhl", "red wings": "nhl", "panthers": "nhl",
    "canadiens": "nhl", "senators": "nhl", "lightning": "nhl", "maple leafs": "nhl",
    "hurricanes": "nhl", "blue jackets": "nhl", "devils": "nhl", "islanders": "nhl",
    "rangers": "nhl", "flyers": "nhl", "penguins": "nhl",
    "blackhawks": "nhl", "avalanche": "nhl", "stars": "nhl", "wild": "nhl",
    "predators": "nhl", "blues": "nhl", "jets": "nhl", "ducks": "nhl",
    "flames": "nhl", "oilers": "nhl", "kings": "nhl", "sharks": "nhl",
    "kraken": "nhl", "canucks": "nhl", "golden knights": "nhl", "capitals": "nhl",
    "coyotes": "nhl",
}

def detect_sport(match_query: str) -> str:
    query_lower = match_query.lower()
    if "tennis" in query_lower or " atp " in query_lower or " wta " in query_lower:
        return "tennis"
    if "mma" in query_lower or "ufc" in query_lower:
        return "mma"
    if "boxing" in query_lower:
        return "boxing"
    if "f1" in query_lower or "grand prix" in query_lower or "formula 1" in query_lower:
        return "f1"
    for sport in NO_DRAW_SPORTS:
        if sport in query_lower:
            return sport
    for team, sport in TEAM_SPORT_MAP.items():
        if team in query_lower:
            return sport
    if " vs " in query_lower or " vs. " in query_lower:
        return "soccer"
    return "unknown"

# ── LLM PROMPT TEMPLATES ────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a sports probability matrix engine. "
    "Output ONLY valid JSON. Do not include markdown, code blocks, or any text outside the JSON. "
    "Never reference betting odds, bookmaker lines, or market prices. "
    "Base probabilities solely on fundamental match data. "
    "BE CONCISE: every string value must be 1 short sentence or phrase — never paragraphs."
    "DO NOT output reasoning, analysis, or thinking — ONLY the JSON object."
)

TEAM_SPORT_USER_PROMPT = (
    'Match: "{match}"\n'
    "Sport: {sport}\n"
    "Draws possible: {draws_possible}\n"
    "Analyze using only: recent form (last 6-10), xG trends, H2H (venue-adjusted), "
    "injuries/suspensions, tactical matchup, home advantage, fatigue/rest days.\n"
    "Return JSON:\n"
    "{{\n"
    '  "match_name": str,\n'
    '  "home_team": str,\n'
    '  "away_team": str,\n'
    '  "true_probabilities": {{"home_win": float, "draw": float, "away_win": float}},\n'
    '  "matrix_factors": {{\n'
    '    "home_form": str, "away_form": str,\n'
    '    "home_absences": str, "away_absences": str,\n'
    '    "tactical_edge": str\n'
    '  }},\n'
    '  "forecast": str,\n'
    '  "confidence": float\n'
    "}}\n"
    "{draw_rule}"
    "home_win+draw+away_win MUST sum to 1.0.\n"
    "RULES:\n"
    "- Every string value must be 1 short sentence or phrase. NEVER write paragraphs.\n"
    "- forecast: 1 sentence max, 15 words max.\n"
    "- form/absences/tactical_edge: 1 phrase each, 8 words max.\n"
    "- match_name/home_team/away_team: team names only, no extra words.\n"
    "- DO NOT output any reasoning, analysis, or thinking text. ONLY the JSON object."
)

INDIVIDUAL_SPORT_USER_PROMPT = (
    'Match: "{match}"\n'
    "Sport: {sport} (individual sport — no draws, no home/away teams)\n"
    "Draws possible: no (set draw to 0.0)\n"
    "Analyze using only: recent form (last 6-10 matches), head-to-head record, "
    "surface/court preference (tennis), injuries, fatigue/rest days, "
    "tournament stage pressure, playing style matchup.\n"
    "Return JSON:\n"
    "{{\n"
    '  "match_name": str,\n'
    '  "home_team": str (use first player/competitor name),\n'
    '  "away_team": str (use second player/competitor name),\n'
    '  "true_probabilities": {{"home_win": float, "draw": 0.0, "away_win": float}},\n'
    '  "matrix_factors": {{\n'
    '    "home_form": str, "away_form": str,\n'
    '    "home_absences": str, "away_absences": str,\n'
    '    "tactical_edge": str\n'
    '  }},\n'
    '  "forecast": str,\n'
    '  "confidence": float\n'
    "}}\n"
    "IMPORTANT: {sport} does NOT have draws. Set draw to 0.0. home_win+away_win MUST sum to 1.0.\n"
    "RULES:\n"
    "- This is an INDIVIDUAL sport — DO NOT think about teams, venues, or home advantage.\n"
    "- 'home_team' = first player/competitor listed, 'away_team' = second player/competitor.\n"
    "- Every string value must be 1 short sentence or phrase. NEVER write paragraphs.\n"
    "- forecast: 1 sentence max, 15 words max.\n"
    "- form/absences/tactical_edge: 1 phrase each, 8 words max.\n"
    "- match_name: just the two player/competitor names, no extra words.\n"
    "- DO NOT output reasoning or analysis text — ONLY the JSON object."
)

def build_user_prompt(match_query: str, sport: str = None) -> str:
    if sport is None:
        sport = detect_sport(match_query)
    if is_individual_sport(sport):
        return INDIVIDUAL_SPORT_USER_PROMPT.format(match=match_query, sport=sport.upper())
    else:
        draws_possible = sport_has_draws(sport)
        if draws_possible:
            draw_rule = ""
        else:
            draw_rule = (
                f"IMPORTANT: {sport.upper()} does NOT have draws. "
                "Set draw to 0.0. Only home_win and away_win should sum to 1.0.\n"
            )
        return TEAM_SPORT_USER_PROMPT.format(
            match=match_query, sport=sport.upper(),
            draws_possible="yes" if draws_possible else "no", draw_rule=draw_rule,
        )

# ── strict llm output schema (validation) ───────────────────
REQUIRED_KEYS = {
    "match_name", "home_team", "away_team",
    "true_probabilities", "matrix_factors", "forecast", "confidence"
}
PROB_KEYS = {"home_win", "draw", "away_win"}
FACTOR_KEYS = {"home_form", "away_form", "home_absences", "away_absences", "tactical_edge"}

def _strip_control_chars(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

def _strip_trailing_commas(json_str: str) -> str:
    return re.sub(r',\s*([]}])', r'\1', json_str)

def _normalize_double_braces(text: str) -> str:
    text = text.replace("{{", "{").replace("}}", "}")
    return text

def extract_json(raw: str) -> str:
    if not raw:
        raise ValueError("empty response from llm")
    raw = _strip_control_chars(raw)
    raw = raw.strip().lstrip("\ufeff\u200b\u200c\u200d\u2060")
    raw = _normalize_double_braces(raw)

    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return _strip_trailing_commas(inner)

    json_blocks = []
    for m in re.finditer(r'\{', raw):
        start = m.start()
        depth = 0
        for i in range(start, len(raw)):
            ch = raw[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    json_blocks.append(raw[start:i+1])
                    break

    if json_blocks:
        for block in reversed(json_blocks):
            candidate = _strip_trailing_commas(block)
            if '"match_name"' in candidate or '"true_probabilities"' in candidate:
                return candidate
        return _strip_trailing_commas(json_blocks[-1])

    bare = raw.strip().lstrip(',').strip()
    if '"' in bare and ':' in bare:
        wrapped = "{" + bare + "}"
        return _strip_trailing_commas(wrapped)

    raise ValueError(f"cannot extract json from: {raw[:500]}")

def _try_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return json.loads(json.dumps(result))
    except (ValueError, SyntaxError, TypeError):
        pass
    raise ValueError(f"cannot parse as json or python dict. text[:500]: {text[:500]}")

def validate_llm_output(data: dict, sport: str) -> dict:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"missing top-level keys: {missing}")
    tp = data["true_probabilities"]
    missing_p = PROB_KEYS - set(tp.keys())
    if missing_p:
        raise ValueError(f"missing probability keys: {missing_p}")
    if is_individual_sport(sport):
        tp["draw"] = 0.0
        total = tp["home_win"] + tp["away_win"]
        if abs(total - 1.0) > 0.015:
            raise ValueError(f"probabilities sum to {total:.4f}, expected 1.0 (individual sport, no draw)")
    else:
        total = tp["home_win"] + tp["draw"] + tp["away_win"]
        if abs(total - 1.0) > 0.015:
            raise ValueError(f"probabilities sum to {total:.4f}, expected 1.0")
    if not sport_has_draws(sport):
        tp["draw"] = 0.0
        hw_aw = tp["home_win"] + tp["away_win"]
        if hw_aw > 0:
            tp["home_win"] = tp["home_win"] / hw_aw
            tp["away_win"] = tp["away_win"] / hw_aw
    mf = data["matrix_factors"]
    missing_f = FACTOR_KEYS - set(mf.keys())
    if missing_f:
        raise ValueError(f"missing factor keys: {missing_f}")
    if not isinstance(data["confidence"], (int, float)):
        raise ValueError("confidence must be numeric")
    return data

# ── api client ──────────────────────────────────────────────

def get_surplus_client() -> AsyncOpenAI:
    if not SURPLUS_API_KEY:
        raise Exception("SURPLUS_API_KEY not set")
    return AsyncOpenAI(api_key=SURPLUS_API_KEY, base_url=SURPLUS_BASE_URL)

async def fetch_true_probabilities(match_query: str, sport: str = None) -> dict:
    client = get_surplus_client()
    if sport is None:
        sport = detect_sport(match_query)
    user_prompt = build_user_prompt(match_query, sport)
    response = await client.chat.completions.create(
        model=SURPLUS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    msg = response.choices[0].message
    raw = msg.content or ""

    if raw and '{' not in raw[:200]:
        reasoning = getattr(msg, 'reasoning_content', None) or ''
        if reasoning and '{' in reasoning:
            print(f"[LLM] Content is reasoning prose, using reasoning_content instead", file=sys.stderr)
            raw = reasoning

    if not raw or '{' not in raw:
        tool_calls = getattr(msg, 'tool_calls', None) or []
        if tool_calls:
            for tc in tool_calls:
                func = getattr(tc, 'function', None)
                if func:
                    args = getattr(func, 'arguments', None) or ''
                    if args and '{' in args:
                        raw = args
                        break

    if not raw:
        raise ValueError("empty response from llm")

    print(f"\nLLM RAW ({SURPLUS_MODEL}): {repr(raw)[:2000]}\n", file=sys.stderr)
    json_str = extract_json(raw)
    data = _try_json_parse(json_str)
    return validate_llm_output(data, sport)

# ── edge calculation ────────────────────────────────────────

def calculate_edges(true_probs: dict, odds_home: float = None, odds_draw: float = None, odds_away: float = None):
    if not odds_home or not odds_away:
        return None
    raw_h = 1.0 / odds_home
    raw_d = (1.0 / odds_draw) if odds_draw else 0.0
    raw_a = 1.0 / odds_away
    total = raw_h + raw_d + raw_a
    imp_h = raw_h / total
    imp_d = raw_d / total if odds_draw else 0.0
    imp_a = raw_a / total
    return {
        "implied": {"home": imp_h, "draw": imp_d, "away": imp_a},
        "edges": {
            "home": true_probs["home_win"] - imp_h,
            "draw": (true_probs.get("draw", 0.0) - imp_d) if odds_draw else None,
            "away": true_probs["away_win"] - imp_a
        },
        "overround_pct": (total - 1.0) * 100
    }

# ── discord button view for !match ──────────────────────────

class MatchActionView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=180)

    @discord.ui.button(label="📊 Open Positions", style=discord.ButtonStyle.secondary, custom_id="btn_open_positions")
    async def positions_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        try:
            output = await fetch_positions_output()
            await send_positions_embeds(interaction.followup, "Bullpen Polymarket Positions", output)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to fetch positions: {e}", ephemeral=True)

# ── discord commands ────────────────────────────────────────

@bot.command(name="open", aliases=["positions", "p"])
async def cmd_open(ctx, *, args: str = ""):
    async with ctx.typing():
        addr = None
        src = None
        if "--address" in args:
            m = re.search(r"--address\s+([^\s]+)", args)
            if m:
                addr = m.group(1)
        if "--source" in args:
            m = re.search(r"--source\s+([^\s]+)", args)
            if m:
                src = m.group(1)
        output = await fetch_positions_output(addr, src)
        await send_positions_embeds(ctx, "Bullpen Polymarket Positions", output, addr, src)

@bot.command(name="watchbullpen")
async def cmd_watchbullpen(ctx, action: str = "status", *, args: str = ""):
    global bullpen_watch_state
    act = action.lower()
    if act == "stop":
        bullpen_watch_state["active"] = False
        save_watcher_config(bullpen_watch_state)
        await ctx.send("🛑 Stopped watching Bullpen positions.")
        return
    if act == "status":
        target_ch = bullpen_watch_state.get('channel_id') or MONITOR_CHANNEL_ID
        is_active = "🟢 Active (24/7 Monitoring)" if bullpen_watch_state.get("active") else "🔴 Stopped"
        sl_val = bullpen_watch_state.get("stop_loss")
        sl_str = f"{sl_val}%" if sl_val is not None else "None"
        auto_close_str = "🟢 Enabled (Auto-Sell)" if bullpen_watch_state.get("auto_close_sl", True) else "🔴 Disabled (Alert Only)"
        status_str = (
            f"**Status:** {is_active}\n"
            f"**Channel:** <#{target_ch}>\n"
            f"**Address:** `{bullpen_watch_state.get('address') or 'Default'}`\n"
            f"**Source:** `{bullpen_watch_state.get('source') or 'Default'}`\n"
            f"**Stop Loss:** `{sl_str}`\n"
            f"**Auto Close Trade:** {auto_close_str}\n"
            f"**Interval:** `{bullpen_watch_state.get('interval', 2)}s`\n"
            f"**Periodic Report:** Every 5 min"
        )
        await ctx.send(embed=discord.Embed(title="Bullpen Watcher Status", description=status_str, color=discord.Color.blue()))
        return
    if act == "start":
        addr = None
        src = None
        sl = None
        interval = 2
        target_ch = MONITOR_CHANNEL_ID
        if "--channel" in args:
            m = re.search(r"--channel\s+(\d+)", args)
            if m: target_ch = int(m.group(1))
        elif ctx.channel.id:
            target_ch = ctx.channel.id
        if "--address" in args:
            m = re.search(r"--address\s+([^\s]+)", args)
            if m: addr = m.group(1)
        if "--source" in args:
            m = re.search(r"--source\s+([^\s]+)", args)
            if m: src = m.group(1)
        if "--sl" in args:
            m = re.search(r"--sl\s+(\d+(?:\.\d+)?)", args)
            if m: sl = float(m.group(1))
        if "--interval" in args:
            m = re.search(r"--interval\s+(\d+)", args)
            if m: interval = int(m.group(1))
        bullpen_watch_state["active"] = True
        bullpen_watch_state["channel_id"] = target_ch
        bullpen_watch_state["address"] = addr
        bullpen_watch_state["source"] = src
        bullpen_watch_state["stop_loss"] = sl
        bullpen_watch_state["auto_close_sl"] = True
        bullpen_watch_state["interval"] = max(1, interval)
        bullpen_watch_state["sl_alert_fired"] = False
        save_watcher_config(bullpen_watch_state)
        sl_desc = f"{sl}%" if sl is not None else "Disabled"
        msg = (
            f"🟢 **Started 24/7 background watching of Bullpen positions!**\n"
            f"• Target Channel: <#{target_ch}>\n"
            f"• Monitoring Interval: `{interval}s` (SL check)\n"
            f"• Stop Loss Trigger: `{sl_desc}`\n"
            f"• Auto-Close Action: ⚡ Direct `bullpen polymarket sell` with exact on-chain market tickers\n"
            f"• Periodic Report: Every 5 minutes\n"
            f"• Quick fetch: Type `open` or `!open` anytime to push current positions up."
        )
        await ctx.send(msg)
        return
    await ctx.send("Usage: `watchbullpen start|stop|status [--sl <pct>]` or `!watchbullpen start|stop|status`")

@bot.command(name="minedge")
async def cmd_set_min_edge(ctx, edge_pct: float):
    gid = ctx.guild.id if ctx.guild else ctx.author.id
    set_min_edge(gid, edge_pct)
    await ctx.send(f"min edge → {edge_pct*100:.1f}%")

@bot.command(name="match")
async def cmd_match(ctx, *, args: str):
    async with ctx.typing():
        match_query = args
        oh = od = oa = None
        polymarket_slug = None
        url_category = None
        gamma_event = None
        detected_sport = None

        # ── detect polymarket url (supports /event/ AND /sports/<cat>/ paths) ──
        slug_from_url, cat_from_url = extract_slug_and_category_from_url(match_query)
        if slug_from_url:
            polymarket_slug = slug_from_url
            url_category = cat_from_url
            print(f"[!match] Polymarket URL detected, slug: {polymarket_slug}, category: {url_category}", file=sys.stderr)

            # ── HIGHEST CONFIDENCE: sport from URL path /sports/<category>/ ──
            if url_category:
                url_sport = detect_sport_from_url_category(url_category)
                if url_sport:
                    detected_sport = url_sport
                    print(f"[!match] Sport from URL category '{url_category}': {detected_sport}", file=sys.stderr)

            gamma_event = await asyncio.to_thread(fetch_event_from_gamma, polymarket_slug)
            if gamma_event:
                # Only use Gamma sport if URL category didn't already give us one
                if detected_sport is None:
                    detected_sport = detect_sport_from_gamma_event(gamma_event)
                    print(f"[!match] Gamma sport detected: {detected_sport}", file=sys.stderr)
                else:
                    print(f"[!match] Using URL category sport ({detected_sport}), Gamma sport: {detect_sport_from_gamma_event(gamma_event)}", file=sys.stderr)

                title = gamma_event.get("title", "")
                teams = gamma_event.get("teams", [])
                if len(teams) >= 2:
                    home_team = teams[0].get("name", "")
                    away_team = teams[1].get("name", "")
                    match_query = f"{home_team} vs. {away_team}"
                elif title:
                    match_query = title.replace(" vs. ", " vs ")
                print(f"[!match] Gamma event resolved: {match_query} (slug: {polymarket_slug}, sport: {detected_sport})", file=sys.stderr)
            else:
                await ctx.send(f"❌ Could not find event for slug `{polymarket_slug}` on Polymarket.")
                return

        # ── parse odds if provided ───────────────────────────
        if "|" in match_query:
            parts = match_query.split("|")
            match_query = parts[0].strip()
            odds_str = parts[1].replace("odds:", "").strip()
            try:
                vals = [float(x.strip()) for x in odds_str.split(",")]
                if len(vals) == 3:
                    oh, od, oa = vals
                elif len(vals) == 2:
                    oh, oa = vals
            except ValueError:
                await ctx.send("bad odds format. use: !match Team A vs Team B | odds: 2.10, 3.40, 3.20")
                return

        gid = ctx.guild.id if ctx.guild else ctx.author.id
        min_edge = get_min_edge(gid)

        if detected_sport is None:
            detected_sport = detect_sport(match_query)
        has_draws = sport_has_draws(detected_sport)
        individual = is_individual_sport(detected_sport)

        try:
            data = await fetch_true_probabilities(match_query, detected_sport)
        except APIStatusError as e:
            await ctx.send(
                f"**Surplus API error** (status {e.status_code}):\n"
                f"```\n{e.message}\n```\n"
                f"**Check:** model name `{SURPLUS_MODEL}`, base URL `{SURPLUS_BASE_URL}`, API key valid?"
            )
            return
        except Exception as e:
            await ctx.send(
                f"**Error**: {type(e).__name__}: {e}\n"
                f"Try again — if this persists, check the model or API key."
            )
            return

        tp  = data["true_probabilities"]
        mf  = data["matrix_factors"]
        fc  = data.get("forecast", "N/A")
        src = f"Surplus ({SURPLUS_MODEL})"

        sport_display = detected_sport.upper()
        if gamma_event:
            sport_obj = gamma_event.get("sport", {})
            if isinstance(sport_obj, dict):
                series_slug = gamma_event.get("seriesSlug", "")
                if series_slug:
                    sport_display = series_slug.upper().replace("-", " ")
            tags = gamma_event.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, dict):
                        label = tag.get("label", "")
                        if label.lower() in ("soccer", "k-league", "premier league", "la liga",
                                             "serie a", "bundesliga", "ligue 1", "mls",
                                             "champions league", "europa league",
                                             "tennis", "atp", "wta", "mma", "ufc", "boxing",
                                             "formula 1", "f1", "allsvenskan"):
                            sport_display = label.upper()
                            break

        if individual:
            home_label = data.get('home_team', 'Player 1')
            away_label = data.get('away_team', 'Player 2')
        else:
            home_label = data.get('home_team', 'Home')
            away_label = data.get('away_team', 'Away')

        embed = discord.Embed(
            title=f"Matrix: {data.get('match_name', match_query)} ({sport_display})",
            color=discord.Color.blue()
        )

        if polymarket_slug:
            embed.add_field(name="Polymarket Slug", value=f"`{polymarket_slug}`", inline=False)
            if gamma_event:
                markets = gamma_event.get("markets", [])
                if markets:
                    market_links = []
                    for m in markets:
                        m_slug = m.get("slug", "")
                        question = m.get("question", "")
                        price = ""
                        prices = m.get("outcomePrices", "[]")
                        try:
                            price_list = json.loads(prices)
                            if price_list and len(price_list) > 0:
                                price = f" — {float(price_list[0])*100:.0f}¢"
                        except (json.JSONDecodeError, ValueError):
                            pass
                        market_links.append(f"[{question}{price}](https://polymarket.com/event/{polymarket_slug}#{m_slug})")
                    embed.add_field(name="Markets", value=_trunc("\n".join(market_links), 1000), inline=False)

        # Build probabilities text — NEVER show a Draw line for no-draw sports
        probs_text = f"**{home_label}**: {tp['home_win']*100:.1f}%\n"
        if has_draws:
            probs_text += f"**Draw**: {tp.get('draw',0)*100:.1f}%\n"
        probs_text += f"**{away_label}**: {tp['away_win']*100:.1f}%"
        embed.add_field(name="True Probabilities (no odds bias)", value=_trunc(probs_text), inline=False)
        embed.add_field(name="Forecast", value=_trunc(fc), inline=False)

        if individual:
            factor_lines = [
                f"{home_label} form: {mf.get('home_form','?')}",
                f"{away_label} form: {mf.get('away_form','?')}",
                f"{home_label} issues: {mf.get('home_absences','none')}",
                f"{away_label} issues: {mf.get('away_absences','none')}",
                f"Style edge: {mf.get('tactical_edge','none')}"
            ]
        else:
            factor_lines = [
                f"Home form: {mf.get('home_form','?')}",
                f"Away form: {mf.get('away_form','?')}",
                f"Home absences: {mf.get('home_absences','none')}",
                f"Away absences: {mf.get('away_absences','none')}",
                f"Tactical edge: {mf.get('tactical_edge','none')}"
            ]
        embed.add_field(name="Factors", value=_trunc("\n".join(factor_lines)), inline=False)

        if oh and oa:
            calc = calculate_edges(tp, oh, od, oa)
            edges   = calc["edges"]
            implied = calc["implied"]
            edge_lines = [f"{home_label} implied: {implied['home']*100:.1f}%  |  edge: {edges['home']*100:+.1f}%"]
            if has_draws and od:
                edge_lines.append(f"Draw implied: {implied['draw']*100:.1f}%  |  edge: {edges['draw']*100:+.1f}%")
            edge_lines.append(f"{away_label} implied: {implied['away']*100:.1f}%  |  edge: {edges['away']*100:+.1f}%")
            embed.add_field(name="Market Edge", value=_trunc("\n".join(edge_lines)), inline=False)
            bets = []
            if edges["home"] >= min_edge:
                bets.append(f"{home_label}: {edges['home']*100:+.1f}%")
            if has_draws and od and edges.get("draw") and edges["draw"] >= min_edge:
                bets.append(f"Draw: {edges['draw']*100:+.1f}%")
            if edges["away"] >= min_edge:
                bets.append(f"{away_label}: {edges['away']*100:+.1f}%")
            if bets:
                embed.add_field(name=f"VALUE (min edge ≥ {min_edge*100:.1f}%)", value=_trunc("\n".join(f"✅ {b}" for b in bets)), inline=False)
            else:
                embed.add_field(name=f"No value (threshold {min_edge*100:.1f}%)", value="Pass — no edge exceeds threshold.", inline=False)

        embed.set_footer(text=f"Engine: {src}  |  min edge: {min_edge*100:.1f}%  |  confidence: {data.get('confidence','?')}")
        view = MatchActionView()
        await ctx.send(embed=embed, view=view)

# ── event listeners ─────────────────────────────────────────

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    content_raw = message.content.strip()
    content_lower = content_raw.lower()
    if content_lower in ("open", "positions"):
        ctx = await bot.get_context(message)
        await cmd_open(ctx)
        return
    if content_lower.startswith("watchbullpen"):
        message.content = "!" + content_raw
    await bot.process_commands(message)

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing argument. Usage: `watchbullpen start --sl 25` or `!{ctx.command.name} <args>`")
        return
    if isinstance(error, commands.BadArgument):
        await ctx.send(f"❌ Bad argument format. Check your inputs and try again.")
        return
    print(f"[ERROR] {ctx.command}: {error}", file=sys.stderr)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    try:
        await ctx.send(f"❌ **Command Failed**: `{type(error).__name__}`\nDetails: {error}")
    except Exception:
        pass

# ── lifecycle ───────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"online: {bot.user.name} ({bot.user.id})")
    print(f"model: {SURPLUS_MODEL}  |  base_url: {SURPLUS_BASE_URL}")
    if bullpen_watch_state.get("task") is None:
        bullpen_watch_state["task"] = asyncio.create_task(bullpen_watcher_loop())
    if bullpen_watch_state.get("active"):
        target_ch = bullpen_watch_state.get('channel_id') or MONITOR_CHANNEL_ID
        print(f"[INFO] bullpen watcher auto-started from config for channel {target_ch}")

@bot.event
async def on_disconnect():
    print("[WARN] disconnected — will auto-reconnect", file=sys.stderr)

@bot.event
async def on_resumed():
    print("[INFO] discord session resumed", file=sys.stderr)

async def main():
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("FATAL: DISCORD_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    async with bot:
        await bot.start(token)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def shutdown():
        print("[INFO] shutting down...", file=sys.stderr)
        for task in asyncio.all_tasks(loop):
            task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print("[INFO] bot stopped cleanly", file=sys.stderr)