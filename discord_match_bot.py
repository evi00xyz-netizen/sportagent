import os
import json
import re
import discord
from discord.ext import commands
from openai import AsyncOpenAI

# ── env config ──────────────────────────────────────────────
DEFAULT_MIN_EDGE = float(os.getenv("MIN_EDGE", "0.05"))

# model routing: set LLM_PROVIDER to "surplus" or "openai" (default: surplus if SURPLUS_API_KEY is set)
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "surplus" if os.getenv("SURPLUS_API_KEY") else "openai")
LLM_MODEL     = os.getenv("LLM_MODEL", "glm-5.2" if LLM_PROVIDER == "surplus" else "gpt-4o")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.surplusintelligence.ai/min30/v1" if LLM_PROVIDER == "surplus" else None)
LLM_API_KEY   = os.getenv("LLM_API_KEY") or os.getenv("SURPLUS_API_KEY") or os.getenv("OPENAI_API_KEY")

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
    "Output ONLY valid JSON matching the schema. "
    "Never use or reference betting odds, bookmaker lines, or market prices. "
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

def extract_json(raw: str) -> str:
    """bulletproof json extraction from any llm output."""
    if not raw:
        raise ValueError("empty response from llm")

    # strip all leading/trailing whitespace including newlines
    raw = raw.strip()

    # try markdown code block: ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # find first { and last } — strip everything before/after
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1].strip()

    # last resort: try to find any json-like structure
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        return m.group(0).strip()

    raise ValueError(f"no json found in llm response. raw (first 300 chars): {raw[:300]}")

def validate_llm_output(data: dict) -> dict:
    """strict output validation — raises on schema violation."""
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

def parse_llm_response(raw: str) -> dict:
    """extract json from llm response and validate it."""
    json_str = extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"invalid json from llm: {e}\n"
            f"extracted json string (first 500 chars): {json_str[:500]}"
        )
    return validate_llm_output(data)

# ── api client ──────────────────────────────────────────────

def get_llm_client() -> AsyncOpenAI:
    """returns an openai-compatible client for whichever provider is configured."""
    if not LLM_API_KEY:
        raise Exception("no API key set — set LLM_API_KEY, SURPLUS_API_KEY, or OPENAI_API_KEY")
    kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return AsyncOpenAI(**kwargs)

async def fetch_true_probabilities(match_query: str) -> dict:
    client = get_llm_client()
    kwargs = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(match=match_query)}
        ],
        "temperature": 0.1,
        "max_tokens": 600,
    }
    # only openai supports response_format json_object — surplus/others may not
    if LLM_PROVIDER == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content
    return parse_llm_response(raw)

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
    """
    !match Arsenal vs Chelsea
    !match Arsenal vs Chelsea | odds: 2.10, 3.40, 3.20
    """
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
        except Exception as e:
            await ctx.send(f"error: {e}")
            return

        tp  = data["true_probabilities"]
        mf  = data["matrix_factors"]
        fc  = data.get("forecast", "N/A")
        src = f"{LLM_PROVIDER} ({LLM_MODEL})"

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
    print(f"provider: {LLM_PROVIDER}  |  model: {LLM_MODEL}  |  base_url: {LLM_BASE_URL or 'default'}")

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("DISCORD_BOT_TOKEN not set")
    else:
        bot.run(token)
