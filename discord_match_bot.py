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
    """Finds the absolute path to the valid bullpen executable."""
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

    fallback_paths = [
        "/usr/local/bin/bullpen",
        "/usr/bin/bullpen"
    ]
    for p in fallback_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    return "bullpen"

def run_bullpen_positions(address: str = None, source: str = None, json_mode: bool = False) -> str:
    """Executes the `bullpen polymarket positions` CLI command via subprocess."""
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

def execute_bullpen_sell(market_slug: str, outcome: str, shares: float) -> str:
    """Executes `bullpen polymarket sell <slug> <outcome> <shares>` with exact market slug."""
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
        err_out = (e.stderr.strip() or e.stdout.strip())

        # Check if bullpen returned "Did you mean:" suggestions as instant safety fallback
        suggested_slugs = re.findall(r'\(([^)]+)\)', err_out)
        if suggested_slugs:
            for suggested in suggested_slugs:
                if "-" in suggested and suggested != market_slug:
                    print(f"[Sell Instant Retry] Retrying with suggested ticker: {suggested}", file=sys.stderr)
                    retry_parts = [bin_path, "polymarket", "sell", suggested, outcome, str(shares)]
                    try:
                        res_retry = subprocess.run(retry_parts, capture_output=True, text=True, check=True, timeout=15, env=env)
                        return res_retry.stdout.strip()
                    except Exception as retry_err:
                        print(f"[Sell Retry Failed] {retry_err}", file=sys.stderr)

        return f"[Sell Error] Exit code {e.returncode}: {err_out}"
    except Exception as e:
        return f"[Sell Error] Failed to sell position: {e}"

_slug_cache = {}

def to_slug(text: str) -> str:
    """Converts a market question string into a clean slug, collapsing multiple hyphens."""
    s = text.lower().replace("…", "").replace("...", "").replace(" ", "-")
    s = re.sub(r'[^a-z0-9\-]+', '-', s)
    s = re.sub(r'-+', '-', s)
    return s.strip('-')

def fetch_exact_slug_by_id(token_or_condition_id: str) -> str:
    """Queries Polymarket Gamma API by asset/token ID, condition ID, or market ID to get the exact ticker/slug."""
    if not token_or_condition_id:
        return None
    if token_or_condition_id in _slug_cache:
        return _slug_cache[token_or_condition_id]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. Query /markets?clob_token_ids=
    try:
        url = f"https://gamma-api.polymarket.com/markets?clob_token_ids={token_or_condition_id}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    m = data[0]
                    exact_slug = m.get("marketSlug") or m.get("slug")
                    if exact_slug:
                        clean_slug = re.sub(r'-+', '-', exact_slug.strip())
                        _slug_cache[token_or_condition_id] = clean_slug
                        return clean_slug
    except Exception as e:
        print(f"[Clob Token Lookup Error] {e}", file=sys.stderr)

    # 2. Query /markets?condition_id=
    try:
        url = f"https://gamma-api.polymarket.com/markets?condition_id={token_or_condition_id}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    m = data[0]
                    exact_slug = m.get("marketSlug") or m.get("slug")
                    if exact_slug:
                        clean_slug = re.sub(r'-+', '-', exact_slug.strip())
                        _slug_cache[token_or_condition_id] = clean_slug
                        return clean_slug
    except Exception as e:
        print(f"[Condition ID Lookup Error] {e}", file=sys.stderr)

    _slug_cache[token_or_condition_id] = None
    return None

def resolve_exact_slug(market_title: str, item_dict: dict = None) -> str:
    """Resolves the exact 100% valid Polymarket ticker/slug for sell execution."""
    if item_dict and isinstance(item_dict, dict):
        for id_key in ["asset_id", "assetId", "tokenId", "token_id", "conditionId", "condition_id", "market_id", "marketId"]:
            val = item_dict.get(id_key)
            if val:
                exact = fetch_exact_slug_by_id(str(val))
                if exact:
                    return exact

        for slug_key in ["eventTicker", "ticker", "marketSlug", "slug", "eventSlug"]:
            raw_slug = item_dict.get(slug_key)
            if raw_slug and isinstance(raw_slug, str) and len(raw_slug.strip()) > 0:
                if not raw_slug.endswith("-") and "..." not in raw_slug and "…" not in raw_slug:
                    return re.sub(r'-+', '-', raw_slug.strip())

    clean_title = market_title.replace("…", "").replace("...", "").strip() if market_title else ""
    return to_slug(clean_title) if clean_title else "unknown-market"

def parse_positions_to_cards(raw_output: str) -> tuple[str, list[dict]]:
    """Parses CLI output (JSON or table) into header text and individual position objects."""
    if not raw_output or raw_output.startswith("[Error]") or raw_output.startswith("[Bullpen Error]"):
        return f"```\n{raw_output}\n```", []

    try:
        data = json.loads(raw_output)
        if isinstance(data, list):
            parsed_positions = []
            total_pnl = 0.0
            for item in data:
                title = item.get("title") or item.get("market") or item.get("question") or "Unknown Market"
                outcome = item.get("outcome") or item.get("outcomeName") or "Yes"

                m_slug = resolve_exact_slug(title, item)

                status = item.get("status", "open")
                shares = str(item.get("shares") or item.get("size") or "0")

                entry_val = item.get("avgPrice") or item.get("entry") or 0.0
                entry = f"${entry_val:.2f}" if isinstance(entry_val, (int, float)) else str(entry_val)

                now_val = item.get("curPrice") or item.get("now") or 0.0
                now = f"${now_val:.2f}" if isinstance(now_val, (int, float)) else str(now_val)

                val_amt = item.get("currentValue") or item.get("value") or 0.0
                val = f"${val_amt:.2f}" if isinstance(val_amt, (int, float)) else str(val_amt)

                pnl_num = item.get("pnl", 0.0)
                if isinstance(pnl_num, (int, float)):
                    pnl_str = f"-${abs(pnl_num):.2f}" if pnl_num < 0 else f"${pnl_num:.2f}"
                    total_pnl += pnl_num
                else:
                    pnl_str = str(pnl_num)

                roe_num = item.get("percentPnl", item.get("roe", 0.0))
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
                slug = resolve_exact_slug(market)

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
    """Sends positions as nicely chunked mobile cards across fields/embeds so no trades are cut off."""
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
    current_embed = discord.Embed(
        title=title,
        description=header,
        color=discord.Color.gold()
    )
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
            current_embed = discord.Embed(
                title=f"{title} (Cont.)",
                color=discord.Color.gold()
            )
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
    """Attempts fetching positions in JSON mode first, falling back to text mode if necessary."""
    res = await asyncio.to_thread(run_bullpen_positions, addr, src, True)
    if res.startswith("[Error]") or res.startswith("[Bullpen Error]") or not res:
        res = await asyncio.to_thread(run_bullpen_positions, addr, src, False)
    return res

async def close_position_task(channel, tp: dict):
    """Executes a single market sell asynchronously in parallel using exact market slug."""
    market_slug = tp['slug']
    close_embed = discord.Embed(
        title="⚡ AUTO-CLOSING POSITION",
        description=f"Executing immediate market sell for `{market_slug}`...",
        color=discord.Color.orange()
    )
    await channel.send(embed=close_embed)

    try:
        shares_num = float(tp['shares'])
        sell_res = await asyncio.to_thread(
            execute_bullpen_sell,
            market_slug,
            tp['outcome'],
            shares_num
        )
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
    """Background task continuously monitoring bullpen positions 24/7 across restarts."""
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
                            await send_positions_embeds(
                                channel,
                                "📊 Bullpen 5-Min Position Report",
                                pos_output,
                                addr,
                                src
                            )
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

# ── sport detection ─────────────────────────────────────────
NO_DRAW_SPORTS = {"mlb", "nba", "nfl", "nhl", "baseball", "basketball", "football", "hockey",
                  "american football", "ice hockey"}

TEAM_SPORT_MAP = {
    # MLB
    "diamondbacks": "mlb", "braves": "mlb", "orioles": "mlb", "red sox": "mlb",
    "cubs": "mlb", "white sox": "mlb", "reds": "mlb", "guardians": "mlb",
    "rockies": "mlb", "tigers": "mlb", "astros": "mlb", "royals": "mlb",
    "angels": "mlb", "dodgers": "mlb", "marlins": "mlb", "brewers": "mlb",
    "twins": "mlb", "mets": "mlb", "yankees": "mlb", "athletics": "mlb",
    "phillies": "mlb", "pirates": "mlb", "padres": "mlb", "giants": "mlb",
    "mariners": "mlb", "cardinals": "mlb", "rays": "mlb", "rangers": "mlb",
    "blue jays": "mlb", "nationals": "mlb",
    # NBA
    "celtics": "nba", "nets": "nba", "knicks": "nba", "76ers": "nba",
    "raptors": "nba", "bulls": "nba", "cavaliers": "nba", "pistons": "nba",
    "pacers": "nba", "bucks": "nba", "hawks": "nba", "hornets": "nba",
    "heat": "nba", "magic": "nba", "wizards": "nba", "nuggets": "nba",
    "timberwolves": "nba", "thunder": "nba", "trail blazers": "nba",
    "jazz": "nba", "warriors": "nba", "clippers": "nba", "lakers": "nba",
    "suns": "nba", "kings": "nba", "mavericks": "nba", "rockets": "nba",
    "grizzlies": "nba", "pelicans": "nba", "spurs": "nba",
    # NFL
    "patriots": "nfl", "dolphins": "nfl", "jets": "nfl", "bills": "nfl",
    "ravens": "nfl", "steelers": "nfl", "browns": "nfl", "bengals": "nfl",
    "texans": "nfl", "colts": "nfl", "jaguars": "nfl", "titans": "nfl",
    "broncos": "nfl", "chiefs": "nfl", "raiders": "nfl", "chargers": "nfl",
    "cowboys": "nfl", "commanders": "nfl", "eagles": "nfl",
    "packers": "nfl", "lions": "nfl", "vikings": "nfl", "bears": "nfl",
    "buccaneers": "nfl", "saints": "nfl", "panthers": "nfl", "falcons": "nfl",
    "rams": "nfl", "49ers": "nfl", "seahawks": "nfl", "cardinals": "nfl",
    # NHL
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
    for sport in NO_DRAW_SPORTS:
        if sport in query_lower:
            if sport in ("baseball",):
                return "mlb"
            if sport in ("basketball",):
                return "nba"
            if sport in ("football", "american football"):
                return "nfl"
            if sport in ("hockey", "ice hockey"):
                return "nhl"
            return sport
    for team, sport in TEAM_SPORT_MAP.items():
        if team in query_lower:
            return sport
    if " vs " in query_lower or " vs. " in query_lower:
        return "soccer"
    return "unknown"

def sport_has_draws(sport: str) -> bool:
    return sport not in ("mlb", "nba", "nfl", "nhl")

# ── strict llm input schema ─────────────────────────────────
SYSTEM_PROMPT = (
    "You are a sports probability matrix engine. "
    "Output ONLY valid JSON. Do not include markdown, code blocks, or any text outside the JSON. "
    "Never reference betting odds, bookmaker lines, or market prices. "
    "Base probabilities solely on fundamental match data. "
    "BE CONCISE: every string value must be 1 short sentence or phrase — never paragraphs."
)

USER_PROMPT_TEMPLATE = (
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
    "- match_name/home_team/away_team: team names only, no extra words."
)

def build_user_prompt(match_query: str) -> str:
    sport = detect_sport(match_query)
    draws_possible = sport_has_draws(sport)
    if draws_possible:
        draw_rule = ""
    else:
        draw_rule = (
            f"IMPORTANT: {sport.upper()} does NOT have draws. "
            "Set draw to 0.0. Only home_win and away_win should sum to 1.0.\n"
        )
    return USER_PROMPT_TEMPLATE.format(
        match=match_query,
        sport=sport.upper(),
        draws_possible="yes" if draws_possible else "no",
        draw_rule=draw_rule
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
    start = raw.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            ch = raw[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return _strip_trailing_commas(raw[start:i+1])
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

async def fetch_true_probabilities(match_query: str) -> dict:
    client = get_surplus_client()
    sport = detect_sport(match_query)
    user_prompt = build_user_prompt(match_query)
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
    if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
        raw = msg.reasoning_content
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
    """Check Polymarket positions using bullpen CLI immediately.
    Usage:
      open / !open / !positions
      open --address 0x123...
      open --source bullpen
    """
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
    """Start/stop watching bullpen positions 24/7 with optional stop-loss trigger.
    Usage:
      !watchbullpen start [--address 0x...] [--source bullpen|polymarket] [--sl 15] [--interval 2] [--channel <id>]
      watchbullpen start --sl 25
      !watchbullpen stop
      !watchbullpen status
    """
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

        if "|" in args:
            parts = args.split("|")
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
        sport = detect_sport(match_query)
        has_draws = sport_has_draws(sport)

        try:
            data = await fetch_true_probabilities(match_query)
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

        embed = discord.Embed(
            title=f"Matrix: {data.get('match_name', match_query)} ({sport.upper()})",
            color=discord.Color.blue()
        )

        probs_text = (
            f"**{data.get('home_team','Home')}**: {tp['home_win']*100:.1f}%\n"
        )
        if has_draws:
            probs_text += f"**Draw**: {tp.get('draw',0)*100:.1f}%\n"
        probs_text += f"**{data.get('away_team','Away')}**: {tp['away_win']*100:.1f}%"

        embed.add_field(
            name="True Probabilities (no odds bias)",
            value=_trunc(probs_text),
            inline=False
        )
        embed.add_field(name="Forecast", value=_trunc(fc), inline=False)

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

            edge_lines = [
                f"Home implied: {implied['home']*100:.1f}%  |  edge: {edges['home']*100:+.1f}%"
            ]
            if has_draws and od:
                edge_lines.append(f"Draw implied: {implied['draw']*100:.1f}%  |  edge: {edges['draw']*100:+.1f}%")
            edge_lines.append(f"Away implied: {implied['away']*100:.1f}%  |  edge: {edges['away']*100:+.1f}%")

            embed.add_field(name="Market Edge", value=_trunc("\n".join(edge_lines)), inline=False)

            bets = []
            if edges["home"] >= min_edge:
                bets.append(f"Home ({data.get('home_team')}): {edges['home']*100:+.1f}%")
            if has_draws and od and edges.get("draw") and edges["draw"] >= min_edge:
                bets.append(f"Draw: {edges['draw']*100:+.1f}%")
            if edges["away"] >= min_edge:
                bets.append(f"Away ({data.get('away_team')}): {edges['away']*100:+.1f}%")

            if bets:
                embed.add_field(
                    name=f"VALUE (min edge ≥ {min_edge*100:.1f}%)",
                    value=_trunc("\n".join(f"✅ {b}" for b in bets)),
                    inline=False
                )
            else:
                embed.add_field(
                    name=f"No value (threshold {min_edge*100:.1f}%)",
                    value="Pass — no edge exceeds threshold.",
                    inline=False
                )

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

# ── global error handler ────────────────────────────────────

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
        await ctx.send(
            f"❌ **Command Failed**: `{type(error).__name__}`\n"
            f"Details: {error}"
        )
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

# ── main with graceful shutdown ─────────────────────────────

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
