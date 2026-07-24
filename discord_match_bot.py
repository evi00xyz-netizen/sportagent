import os
import json
import re
import ast
import sys
import traceback
import discord
from discord.ext import commands
from openai import AsyncOpenAI, APIStatusError

# ── env config ──────────────────────────────────────────────
DEFAULT_MIN_EDGE = float(os.getenv("MIN_EDGE", "0.05"))
SURPLUS_API_KEY  = os.getenv("SURPLUS_API_KEY")
SURPLUS_BASE_URL = os.getenv("SURPLUS_API_URL", "https://api.surplusintelligence.ai/min30/v1")
SURPLUS_MODEL    = os.getenv("SURPLUS_MODEL", "glm-5.2")

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

# ── strict llm input schema ─────────────────────────────────
SYSTEM_PROMPT = (
    "You are a sports probability matrix engine. "
    "Output ONLY valid JSON. Do not include markdown, code blocks, or any text outside the JSON. "
    "Never reference betting odds, bookmaker lines, or market prices. "
    "Base probabilities solely on fundamental match data."
)

USER_PROMPT_TEMPLATE = (
    'Match: "{match}"\n'
    "Analyze using only: recent form (last 6-10), xG trends, H2H (venue-adjusted), "
    "injuries/suspensions, tactical matchup, home advantage, fatigue/rest days.\n"
    "Return JSON:\n"
    "{\n"
    '  "match_name": str,\n'
    '  "home_team": str,\n'
    '  "away_team": str,\n'
    '  "true_probabilities": {"home_win": float, "draw": float, "away_win": float},\n'
    '  "matrix_factors": {\n'
    '    "home_form": str, "away_form": str,\n'
    '    "home_absences": str, "away_absences": str,\n'
    '    "tactical_edge": str\n'
    '  },\n'
    '  "forecast": str,\n'
    '  "confidence": float\n'
    "}\n"
    "home_win+draw+away_win MUST sum to 1.0."
)

# ── strict llm output schema (validation) ───────────────────
REQUIRED_KEYS = {
    "match_name", "home_team", "away_team",
    "true_probabilities", "matrix_factors", "forecast", "confidence"
}
PROB_KEYS = {"home_win", "draw", "away_win"}
FACTOR_KEYS = {"home_form", "away_form", "home_absences", "away_absences", "tactical_edge"}

def _strip_control_chars(s: str) -> str:
    """remove all ascii control characters except space, \\n, \\t."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

def _strip_trailing_commas(json_str: str) -> str:
    """remove trailing commas before closing } or ]."""
    return re.sub(r',\s*([]}])', r'\1', json_str)

def extract_json(raw: str) -> str:
    """aggressive json extraction."""
    if not raw:
        raise ValueError("empty response from llm")

    raw = _strip_control_chars(raw)
    raw = raw.strip().lstrip("\ufeff\u200b\u200c\u200d\u2060")

    # case 1: markdown code block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return _strip_trailing_commas(inner)

    # case 2: find { ... } via brace counting
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

    # case 3: no braces — bare key:value pairs
    bare = raw.strip().lstrip(',').strip()
    if '"' in bare and ':' in bare:
        wrapped = "{" + bare + "}"
        return _strip_trailing_commas(wrapped)

    raise ValueError(f"cannot extract json from: {raw[:500]}")

def _try_json_parse(text: str) -> dict:
    """try json.loads, then ast.literal_eval."""
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

def validate_llm_output(data: dict) -> dict:
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
    response = await client.chat.completions.create(
        model=SURPLUS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(match=match_query)}
        ],
        temperature=0.1,
        max_tokens=600,
    )
    raw = response.choices[0].message.content or ""

    # dump to stderr
    print(f"\nLLM RAW: {repr(raw)[:2000]}\n", file=sys.stderr)

    json_str = extract_json(raw)
    data = _try_json_parse(json_str)
    return validate_llm_output(data)

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

        try:
            data = await fetch_true_probabilities(match_query)
        except APIStatusError as e:
            # surplus API returned an error — show it in full
            await ctx.send(
                f"**Surplus API error** (status {e.status_code}):\n"
                f"```\n{e.message}\n```\n"
                f"**Check:** model name `{SURPLUS_MODEL}`, base URL `{SURPLUS_BASE_URL}`, API key valid?"
            )
            return
        except Exception as e:
            # any other error — include full traceback so we can debug
            tb = traceback.format_exc()
            short_tb = tb[-1500:] if len(tb) > 1500 else tb
            await ctx.send(
                f"**Error**: {type(e).__name__}: {e}\n"
                f"```\n{short_tb}\n```"
            )
            return

        tp  = data["true_probabilities"]
        mf  = data["matrix_factors"]
        fc  = data.get("forecast", "N/A")
        src = f"Surplus ({SURPLUS_MODEL})"

        embed = discord.Embed(
            title=f"Matrix: {data.get('match_name', match_query)}",
            color=discord.Color.blue()
        )

        embed.add_field(
            name="True Probabilities (no odds bias)",
            value=(
                f"**{data.get('home_team','Home')}**: {tp['home_win']*100:.1f}%\n"
                f"**Draw**: {tp.get('draw',0)*100:.1f}%\n"
                f"**{data.get('away_team','Away')}**: {tp['away_win']*100:.1f}%"
            ),
            inline=False
        )
        embed.add_field(name="Forecast", value=fc, inline=False)

        factor_lines = [
            f"Home form: {mf.get('home_form','?')}",
            f"Away form: {mf.get('away_form','?')}",
            f"Home absences: {mf.get('home_absences','none')}",
            f"Away absences: {mf.get('away_absences','none')}",
            f"Tactical edge: {mf.get('tactical_edge','none')}"
        ]
        embed.add_field(name="Factors", value="\n".join(factor_lines), inline=False)

        if oh and oa:
            calc = calculate_edges(tp, oh, od, oa)
            edges   = calc["edges"]
            implied = calc["implied"]

            edge_lines = [
                f"Home implied: {implied['home']*100:.1f}%  |  edge: {edges['home']*100:+.1f}%"
            ]
            if od:
                edge_lines.append(f"Draw implied: {implied['draw']*100:.1f}%  |  edge: {edges['draw']*100:+.1f}%")
            edge_lines.append(f"Away implied: {implied['away']*100:.1f}%  |  edge: {edges['away']*100:+.1f}%")

            embed.add_field(name="Market Edge", value="\n".join(edge_lines), inline=False)

            bets = []
            if edges["home"] >= min_edge:
                bets.append(f"Home ({data.get('home_team')}): {edges['home']*100:+.1f}%")
            if od and edges.get("draw") and edges["draw"] >= min_edge:
                bets.append(f"Draw: {edges['draw']*100:+.1f}%")
            if edges["away"] >= min_edge:
                bets.append(f"Away ({data.get('away_team')}): {edges['away']*100:+.1f}%")

            if bets:
                embed.add_field(
                    name=f"VALUE (min edge ≥ {min_edge*100:.1f}%)",
                    value="\n".join(f"✅ {b}" for b in bets),
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

@bot.event
async def on_ready():
    print(f"online: {bot.user.name} ({bot.user.id})")
    print(f"model: {SURPLUS_MODEL}  |  base_url: {SURPLUS_BASE_URL}")

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("DISCORD_BOT_TOKEN not set")
    else:
        bot.run(token)