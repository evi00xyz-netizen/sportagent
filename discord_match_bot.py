import os
import json
import re
import ast
import sys
import traceback
import asyncio
import signal
import subprocess
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

# discord embed field limit
EMBED_FIELD_MAX = 1024

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
        "interval": 2,
        "notify_on_change": True,
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
            "interval": state.get("interval", 2),
            "notify_on_change": state.get("notify_on_change", True),
            "last_positions": state.get("last_positions")
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(to_save, f, indent=2)
    except Exception as e:
        print(f"[Config Save Error] {e}", file=sys.stderr)

bullpen_watch_state = load_watcher_config()
bullpen_watch_state["task"] = None

def run_bullpen_positions(address: str = None, source: str = None) -> str:
    """Executes the `bullpen polymarket positions` CLI command with timeout."""
    cmd = ["bullpen", "polymarket", "positions"]
    if address:
        cmd.extend(["--address", address])
    if source:
        cmd.extend(["--source", source])
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return res.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Error] `bullpen` CLI command timed out after 10 seconds."
    except FileNotFoundError:
        return "[Error] `bullpen` CLI tool is not installed or not in PATH on this server."
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() or e.stdout.strip()
        return f"[Bullpen Error] {err_msg}"

async def bullpen_watcher_loop():
    """Background task continuously monitoring bullpen positions 24/7 across restarts."""
    while True:
        try:
            if bullpen_watch_state["active"]:
                target_channel_id = bullpen_watch_state.get("channel_id") or MONITOR_CHANNEL_ID
                channel = bot.get_channel(target_channel_id)
                if channel:
                    addr = bullpen_watch_state.get("address")
                    src = bullpen_watch_state.get("source")
                    sl = bullpen_watch_state.get("stop_loss")
                    notify_change = bullpen_watch_state.get("notify_on_change", True)
                    last_pos = bullpen_watch_state.get("last_positions")

                    pos_output = await asyncio.to_thread(run_bullpen_positions, addr, src)
                    
                    # Do not trigger position change alerts or SL on CLI errors
                    if pos_output.startswith("[Error]") or pos_output.startswith("[Bullpen Error]"):
                        print(f"[Bullpen Watcher CLI Warning] {pos_output}", file=sys.stderr)
                    else:
                        # Stop loss check
                        sl_triggered = False
                        if sl is not None:
                            matches = re.findall(r"-\d+(?:\.\d+)?%", pos_output)
                            for m in matches:
                                loss_val = abs(float(m.replace("%", "")))
                                if loss_val >= sl:
                                    sl_triggered = True
                                    break

                        if sl_triggered:
                            embed = discord.Embed(
                                title="🚨 BULLPEN STOP LOSS TRIGGERED",
                                description=f"Stop Loss threshold of `{sl}%` reached/exceeded!",
                                color=discord.Color.red()
                            )
                            embed.add_field(name="Current Positions", value=_trunc(f"```\n{pos_output}\n```"), inline=False)
                            await channel.send(embed=embed)
                        elif notify_change and last_pos and last_pos != pos_output:
                            embed = discord.Embed(
                                title="📊 Bullpen Positions Updated",
                                description="Change detected in monitored positions:",
                                color=discord.Color.blue()
                            )
                            embed.add_field(name="Latest Positions", value=_trunc(f"```\n{pos_output}\n```"), inline=False)
                            await channel.send(embed=embed)

                        if pos_output != last_pos:
                            bullpen_watch_state["last_positions"] = pos_output
                            save_watcher_config(bullpen_watch_state)

        except Exception as e:
            print(f"[Bullpen Watcher Loop Error] {e}", file=sys.stderr)

        interval = max(1, bullpen_watch_state.get("interval", 2))
        await asyncio.sleep(interval)

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

# ── discord commands ────────────────────────────────────────

@bot.command(name="positions")
async def cmd_positions(ctx, *, args: str = ""):
    """Check Polymarket positions using bullpen CLI.
    Usage:
      !positions
      !positions --address 0x123...
      !positions --source bullpen
      !positions --source polymarket
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

        output = await asyncio.to_thread(run_bullpen_positions, addr, src)
        
        embed = discord.Embed(
            title="Bullpen Polymarket Positions",
            description=f"```\n{_trunc(output, 2000)}\n```",
            color=discord.Color.gold()
        )
        if addr:
            embed.add_field(name="Address", value=addr, inline=True)
        if src:
            embed.add_field(name="Source", value=src, inline=True)
        await ctx.send(embed=embed)

@bot.command(name="watchbullpen")
async def cmd_watchbullpen(ctx, action: str = "status", *, args: str = ""):
    """Start/stop watching bullpen positions 24/7 with optional stop-loss trigger.
    Usage:
      !watchbullpen start [--address 0x...] [--source bullpen|polymarket] [--sl 15] [--interval 2] [--channel <id>]
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
        status_str = (
            f"**Status:** {'🟢 Active (24/7 Monitoring)' if bullpen_watch_state.get('active') else '🔴 Stopped'}\n"
            f"**Channel:** <#{target_ch}>\n"
            f"**Address:** `{bullpen_watch_state.get('address') or 'Default'}`\n"
            f"**Source:** `{bullpen_watch_state.get('source') or 'Default'}`\n"
            f"**Stop Loss:** `{f'{bullpen_watch_state.get(\"stop_loss\")}%' if bullpen_watch_state.get('stop_loss') else 'None'}`\n"
            f"**Interval:** `{bullpen_watch_state.get('interval', 2)}s`"
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
        bullpen_watch_state["interval"] = max(1, interval)

        save_watcher_config(bullpen_watch_state)

        msg = (
            f"🟢 **Started 24/7 background watching of Bullpen positions!**\n"
            f"• Target Channel: <#{target_ch}>\n"
            f"• Interval: `{interval}s`\n"
            f"• Address: `{addr or 'Default'}`\n"
            f"• Source: `{src or 'Default'}`\n"
            f"• Stop Loss Trigger: `{f'{sl}%' if sl else 'Disabled'}`\n"
            f"• Configuration saved to disk — will automatically resume across restarts."
        )
        await ctx.send(msg)
        return

    await ctx.send("Usage: `!watchbullpen start|stop|status [options]`")

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
        await ctx.send(embed=embed)

# ── global error handler ────────────────────────────────────

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing argument. Usage: `!{ctx.command.name} <args>`")
        return
    if isinstance(error, commands.BadArgument):
        await ctx.send(f"Bad argument. Check the format and try again.")
        return
    print(f"[ERROR] {ctx.command}: {error}", file=sys.stderr)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    try:
        await ctx.send(
            f"Something went wrong. The bot is still running — try again.\n"
            f"`{type(error).__name__}`"
        )
    except Exception:
        pass

# ── lifecycle ───────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"online: {bot.user.name} ({bot.user.id})")
    print(f"model: {SURPLUS_MODEL}  |  base_url: {SURPLUS_BASE_URL}")
    if bullpen_watch_state.get("task") is None:
        bullpen_watch_state["task"] = asyncio.io_task = asyncio.create_task(bullpen_watcher_loop())
    if bullpen_watch_state.get("active"):
        target_ch = bullpen_watch_state.get('channel_id') or MONITOR_CHANNEL_ID
        print(f"[INFO] bullpen watcher auto-started from config for channel {target_ch}")

@bot.event
async def on_disconnect():
    print("[WARN] discord disconnected — will auto-reconnect", file=sys.stderr)

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
