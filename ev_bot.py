"""
ev_bot.py — EV-Based Polymarket Betting Bot (Flat Bet / Analysis Mode)
======================================================================
Architecture:
  1. Gamma API      → fetch market data + clobTokenIds
  2. OpenAI (gpt-5.4) → Pydantic Structured Outputs for fundamental probabilities
  3. CLOB API       → live best-ask prices → decimal odds
  4. EV Math        → deterministic EV = (prob * odds) - 1
  5. Flat Bet       → displays exact bet sizing, no on-chain execution needed

No private keys, no funder address, no py-clob-client required.
Run independently alongside your existing discord_match_bot.py.
"""

import os
import re
import json
import sys
import asyncio
import signal
import traceback
import urllib.request
import urllib.parse
from typing import Optional

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

import discord
from discord.ext import commands

# ── env config ──────────────────────────────────────────────
DISCORD_TOKEN      = os.getenv("EV_BOT_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")
SURPLUS_API_KEY    = os.getenv("SURPLUS_API_KEY")
SURPLUS_BASE_URL   = os.getenv("SURPLUS_API_URL", "https://api.surplusintelligence.ai/min30/v1")
SURPLUS_MODEL      = os.getenv("SURPLUS_MODEL", "gpt-5.4")
DEFAULT_EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.05"))
DEFAULT_BET_SIZE_USD = float(os.getenv("DEFAULT_BET_SIZE", "10.0"))

# ── Pydantic Structured Output Schema ───────────────────────

class MatchProbabilities(BaseModel):
    home_win: float = Field(..., ge=0.0, le=1.0)
    draw: float = Field(..., ge=0.0, le=1.0)
    away_win: float = Field(..., ge=0.0, le=1.0)
    home_team: str = Field(...)
    away_team: str = Field(...)
    match_name: str = Field(...)
    key_factors: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)

MATCH_PROB_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "match_probabilities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "home_win": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "draw": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "away_win": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "home_team": {"type": "string"},
                "away_team": {"type": "string"},
                "match_name": {"type": "string"},
                "key_factors": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["home_win", "draw", "away_win", "home_team", "away_team", "match_name", "key_factors", "confidence"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are a football probability engine. "
    "Calculate fundamental probabilities based ONLY on football team fundamentals: "
    "Expected Goals (xG), fixture congestion, squad depth, tactical matchups, "
    "injuries, manager quality, home/away form, rest days, and head-to-head history. "
    "You must COMPLETELY IGNORE market consensus, betting volume, current odds, "
    "or any market-derived data. Calculate without market consensus completely. "
    "Output ONLY valid JSON matching the schema exactly. "
    "home_win + draw + away_win MUST sum exactly to 1.0."
)

USER_PROMPT_TEMPLATE = (
    'Analyze this football match: "{match_title}"\n'
    "Home team: {home_team}\n"
    "Away team: {away_team}\n"
    "League/Competition: {league}\n"
    "Match date: {match_date}\n\n"
    "Consider these fundamental factors:\n"
    "- Recent form (last 6-10 matches) and xG trends\n"
    "- Head-to-head record (venue-adjusted)\n"
    "- Injuries, suspensions, and squad availability\n"
    "- Fixture congestion and rest days\n"
    "- Tactical matchup and manager quality\n"
    "- Home/away performance splits\n\n"
    "IMPORTANT: Base your analysis ONLY on football fundamentals. "
    "IGNORE all market data, betting odds, and trading volume. "
    "Output the exact JSON schema with probabilities summing to 1.0."
)

# ── Gamma API ───────────────────────────────────────────────

def fetch_event_from_gamma(slug: str) -> Optional[dict]:
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
    except Exception as e:
        print(f"[Gamma] Event lookup failed for '{slug}': {e}", file=sys.stderr)
    return None


def extract_markets_from_event(event: dict) -> list[dict]:
    markets = event.get("markets", [])
    result = []
    for m in markets:
        question = (m.get("question") or "").lower()
        clob_ids = m.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (json.JSONDecodeError, TypeError):
                clob_ids = []
        if not clob_ids or len(clob_ids) == 0:
            continue
        token_id = clob_ids[0]
        if "draw" in question or "tie" in question:
            market_type = "draw"
        elif "win" in question:
            home_name = ""
            away_name = ""
            teams = event.get("teams", [])
            if len(teams) >= 2:
                home_name = (teams[0].get("name") or "").lower()
                away_name = (teams[1].get("name") or "").lower()
            if home_name and home_name in question:
                market_type = "home"
            elif away_name and away_name in question:
                market_type = "away"
            elif home_name and any(w in question for w in home_name.split()):
                market_type = "home"
            elif away_name and any(w in question for w in away_name.split()):
                market_type = "away"
            else:
                continue
        else:
            continue
        result.append({
            "type": market_type,
            "token_id": token_id,
            "question": m.get("question", ""),
            "slug": m.get("slug", ""),
        })
    return result


def parse_polymarket_url(url: str) -> Optional[str]:
    m = re.search(
        r'https?://polymarket\.com/(?:event|sports/[a-z0-9]+)/([a-z0-9][a-z0-9\-]+[a-z0-9])',
        url, re.IGNORECASE
    )
    return m.group(1) if m else None


# ── CLOB API ────────────────────────────────────────────────

def fetch_clob_best_ask(token_id: str) -> Optional[float]:
    url = f"https://clob.polymarket.com/price?token_id={token_id}&side=buy"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                price_str = data.get("price", "0")
                return float(price_str)
    except Exception as e:
        print(f"[CLOB] Price fetch failed for token {token_id}: {e}", file=sys.stderr)
    return None


def share_price_to_decimal_odds(price: float) -> float:
    if price <= 0:
        return float("inf")
    return 1.0 / price


# ── OpenAI Probability Engine ───────────────────────────────

def get_openai_client() -> AsyncOpenAI:
    if not SURPLUS_API_KEY:
        raise RuntimeError("SURPLUS_API_KEY not set")
    return AsyncOpenAI(api_key=SURPLUS_API_KEY, base_url=SURPLUS_BASE_URL)


async def fetch_fundamental_probabilities(
    match_title: str, home_team: str, away_team: str,
    league: str = "Unknown", match_date: str = "Unknown",
) -> MatchProbabilities:
    client = get_openai_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        match_title=match_title, home_team=home_team, away_team=away_team,
        league=league, match_date=match_date,
    )
    try:
        response = await client.chat.completions.create(
            model=SURPLUS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, max_tokens=1000,
            response_format=MATCH_PROB_SCHEMA,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("empty response from LLM")
        data = json.loads(raw)
        total = data["home_win"] + data["draw"] + data["away_win"]
        if abs(total - 1.0) > 0.001:
            data["home_win"] /= total
            data["draw"] /= total
            data["away_win"] /= total
        return MatchProbabilities(**data)
    except Exception as e:
        print(f"[OpenAI] Structured output failed ({e}), falling back", file=sys.stderr)
        response = await client.chat.completions.create(
            model=SURPLUS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, max_tokens=1000,
        )
        raw = response.choices[0].message.content or ""
        return _parse_raw_probabilities(raw)


def _parse_raw_probabilities(raw: str) -> MatchProbabilities:
    json_str = raw
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL | re.IGNORECASE)
    if m:
        json_str = m.group(1).strip()
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start >= 0 and end > start:
        json_str = json_str[start:end+1]
    data = json.loads(json_str)
    total = data.get("home_win", 0) + data.get("draw", 0) + data.get("away_win", 0)
    if total > 0 and abs(total - 1.0) > 0.001:
        data["home_win"] = data.get("home_win", 0) / total
        data["draw"] = data.get("draw", 0) / total
        data["away_win"] = data.get("away_win", 0) / total
    return MatchProbabilities(
        home_win=data.get("home_win", 0.33), draw=data.get("draw", 0.34),
        away_win=data.get("away_win", 0.33),
        home_team=data.get("home_team", "Home"), away_team=data.get("away_team", "Away"),
        match_name=data.get("match_name", "Unknown"),
        key_factors=data.get("key_factors", "No factors provided"),
        confidence=data.get("confidence", 0.5),
    )


# ── EV Calculation ─────────────────────────────────────────

def calculate_ev(probs: MatchProbabilities, clob_prices: dict) -> dict:
    results = {
        "home": {"prob": probs.home_win, "price": None, "odds": None, "ev": None, "token_id": None},
        "draw": {"prob": probs.draw, "price": None, "odds": None, "ev": None, "token_id": None},
        "away": {"prob": probs.away_win, "price": None, "odds": None, "ev": None, "token_id": None},
    }
    for outcome in ["home", "draw", "away"]:
        price_data = clob_prices.get(outcome)
        if price_data and price_data.get("price") is not None:
            price = price_data["price"]
            results[outcome]["price"] = price
            results[outcome]["token_id"] = price_data.get("token_id")
            if price > 0:
                odds = share_price_to_decimal_odds(price)
                results[outcome]["odds"] = odds
                results[outcome]["ev"] = (results[outcome]["prob"] * odds) - 1.0
    return results


def calculate_flat_bet(ev: float, prob: float, bet_size: float, threshold: float) -> Optional[dict]:
    if ev < threshold:
        return None
    expected_profit = bet_size * ev
    kelly_fraction = ev / 2 if ev > 0 else None
    return {
        "bet_amount": bet_size,
        "expected_profit": expected_profit,
        "expected_roi_pct": ev * 100,
        "kelly_reference": kelly_fraction,
    }


# ── Discord Bot ─────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

ev_thresholds = {}
bet_sizes = {}

def get_ev_threshold(guild_id: int) -> float:
    return ev_thresholds.get(guild_id, DEFAULT_EV_THRESHOLD)

def set_ev_threshold(guild_id: int, threshold: float):
    ev_thresholds[guild_id] = threshold

def get_bet_size(guild_id: int) -> float:
    return bet_sizes.get(guild_id, DEFAULT_BET_SIZE_USD)

def set_bet_size(guild_id: int, size: float):
    bet_sizes[guild_id] = size


@bot.command(name="ev")
async def cmd_ev(ctx, *, args: str = ""):
    """Analyze a Polymarket football event and calculate EV with flat bet sizing."""
    print(f"[EV CMD] Received !ev from {ctx.author} with args: '{args}'", file=sys.stderr)

    if not args:
        help_text = (
            "**EV Bot — Flat Bet Analysis**\n"
            "`!ev <polymarket_url>` — Analyze match, show EV + flat bet sizing\n"
            "`!ev <polymarket_url> --bet <amount>` — Custom bet size\n"
            "`!ev threshold <0.05>` — Set EV threshold (e.g. 0.05 = 5%)\n"
            "`!ev betsize <25>` — Set default flat bet size in USD\n"
            "`!evstatus` — Show current config"
        )
        await ctx.send(help_text)
        return

    try:
        # ── Handle threshold setting ──
        if args.lower().startswith("threshold"):
            try:
                parts = args.split()
                if len(parts) >= 2:
                    new_threshold = float(parts[1])
                    gid = ctx.guild.id if ctx.guild else ctx.author.id
                    set_ev_threshold(gid, new_threshold)
                    await ctx.send(f"✅ EV threshold set to **{new_threshold*100:.1f}%**")
                    return
            except ValueError:
                await ctx.send("❌ Invalid threshold. Use: `!ev threshold 0.05`")
                return

        # ── Handle bet size setting ──
        if args.lower().startswith("betsize"):
            try:
                parts = args.split()
                if len(parts) >= 2:
                    new_size = float(parts[1])
                    gid = ctx.guild.id if ctx.guild else ctx.author.id
                    set_bet_size(gid, new_size)
                    await ctx.send(f"✅ Default flat bet size set to **${new_size:.2f}**")
                    return
            except ValueError:
                await ctx.send("❌ Invalid bet size. Use: `!ev betsize 25`")
                return

        # ── Parse args ──
        custom_bet_size = None
        bet_match = re.search(r'--bet\s+(\d+(?:\.\d+)?)', args)
        if bet_match:
            custom_bet_size = float(bet_match.group(1))
            args = args.replace(bet_match.group(0), "").strip()

        url = args.strip()
        print(f"[EV CMD] URL to analyze: '{url}'", file=sys.stderr)

        # ── Extract slug ──
        slug = parse_polymarket_url(url)
        if not slug:
            await ctx.send(f"❌ Could not extract event slug from URL. Use a full Polymarket event URL like:\n`https://polymarket.com/event/<slug>`")
            return

        print(f"[EV CMD] Extracted slug: '{slug}'", file=sys.stderr)

        async with ctx.typing():
            # ── Step 1: Fetch Gamma event ──
            print(f"[EV CMD] Fetching Gamma event for slug: {slug}", file=sys.stderr)
            event = await asyncio.to_thread(fetch_event_from_gamma, slug)
            if not event:
                await ctx.send(f"❌ Event not found for slug `{slug}`. Check the URL is correct and the event exists on Polymarket.")
                return

            print(f"[EV CMD] Gamma event found: {event.get('title', '?')}", file=sys.stderr)

            title = event.get("title", slug)
            teams = event.get("teams", [])
            home_team = teams[0].get("name", "Home") if len(teams) >= 1 else "Home"
            away_team = teams[1].get("name", "Away") if len(teams) >= 2 else "Away"

            series = event.get("series", [])
            league = "Unknown"
            if isinstance(series, list) and len(series) > 0:
                league = series[0].get("title", "Unknown")
            elif isinstance(event.get("sport"), dict):
                league = event["sport"].get("sport", "Unknown").upper()

            match_date = event.get("startDate") or event.get("scheduledStart") or "Unknown"

            # ── Step 2: Extract markets ──
            markets = extract_markets_from_event(event)
            print(f"[EV CMD] Found {len(markets)} markets", file=sys.stderr)
            if len(markets) < 2:
                await ctx.send(f"❌ Could not find enough markets (home/draw/away). Found: {len(markets)}. This may not be a football match market.")
                return

            market_lookup = {}
            for m in markets:
                market_lookup[m["type"]] = m

            # ── Step 3: Fetch CLOB prices ──
            clob_prices = {}
            for outcome_type in ["home", "draw", "away"]:
                m = market_lookup.get(outcome_type)
                if m:
                    price = await asyncio.to_thread(fetch_clob_best_ask, m["token_id"])
                    clob_prices[outcome_type] = {
                        "price": price, "token_id": m["token_id"],
                        "question": m.get("question", ""),
                    }
                else:
                    clob_prices[outcome_type] = {"price": None, "token_id": None, "question": ""}

            print(f"[EV CMD] CLOB prices: home={clob_prices.get('home',{}).get('price')}, draw={clob_prices.get('draw',{}).get('price')}, away={clob_prices.get('away',{}).get('price')}", file=sys.stderr)

            # ── Step 4: AI Probability Engine ──
            print(f"[EV CMD] Calling AI for: {title}", file=sys.stderr)
            try:
                probs = await fetch_fundamental_probabilities(
                    match_title=title, home_team=home_team, away_team=away_team,
                    league=league, match_date=str(match_date),
                )
            except Exception as e:
                await ctx.send(f"❌ **AI Engine Error**: {type(e).__name__}: {e}\nCheck SURPLUS_API_KEY and model `{SURPLUS_MODEL}`.")
                return

            print(f"[EV CMD] AI probs: home={probs.home_win:.3f}, draw={probs.draw:.3f}, away={probs.away_win:.3f}", file=sys.stderr)

            # ── Step 5: EV Calculation ──
            ev_results = calculate_ev(probs, clob_prices)
            gid = ctx.guild.id if ctx.guild else ctx.author.id
            threshold = get_ev_threshold(gid)
            bet_size = custom_bet_size or get_bet_size(gid)

            # ── Build Discord Embed ──
            embed = discord.Embed(
                title=f"📊 EV Analysis: {probs.match_name}",
                description=f"**League:** {league} | **Date:** {match_date}\n**Slug:** `{slug}`",
                color=discord.Color.blue(),
            )

            prob_lines = [
                f"🏠 **{probs.home_team}**: {probs.home_win*100:.1f}%",
                f"🤝 **Draw**: {probs.draw*100:.1f}%",
                f"🚶 **{probs.away_team}**: {probs.away_win*100:.1f}%",
            ]
            embed.add_field(name="AI Fundamental Probabilities", value="\n".join(prob_lines), inline=False)

            ev_lines = []
            bet_lines = []
            positive_ev_outcomes = []

            for outcome, label, emoji in [
                ("home", probs.home_team, "🏠"),
                ("draw", "Draw", "🤝"),
                ("away", probs.away_team, "🚶"),
            ]:
                r = ev_results[outcome]
                price_str = f"{r['price']*100:.1f}¢" if r["price"] is not None else "N/A"
                odds_str = f"{r['odds']:.2f}" if r["odds"] is not None else "N/A"
                ev_str = f"{r['ev']*100:+.1f}%" if r["ev"] is not None else "N/A"

                if r["ev"] is not None and r["ev"] >= threshold:
                    ev_lines.append(f"✅ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV **{ev_str}**")
                    positive_ev_outcomes.append(outcome)
                    flat = calculate_flat_bet(r["ev"], r["prob"], bet_size, threshold)
                    if flat:
                        bet_lines.append(
                            f"{emoji} **{label}**: Bet `${flat['bet_amount']:.2f}` → "
                            f"Expected profit `${flat['expected_profit']:.2f}` "
                            f"({flat['expected_roi_pct']:+.1f}% ROI)"
                        )
                elif r["ev"] is not None:
                    ev_lines.append(f"❌ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV {ev_str}")
                else:
                    ev_lines.append(f"⚪ {emoji} **{label}**: Price {price_str} | No EV data")

            embed.add_field(
                name=f"CLOB Prices & EV (threshold: {threshold*100:.1f}%)",
                value="\n".join(ev_lines), inline=False,
            )

            if bet_lines:
                embed.add_field(
                    name=f"💰 Flat Bet Sizing (${bet_size:.2f} per +EV outcome)",
                    value="\n".join(bet_lines), inline=False,
                )

            embed.add_field(name="Key Factors", value=probs.key_factors, inline=False)
            embed.add_field(name="AI Confidence", value=f"{probs.confidence*100:.0f}%", inline=True)

            if positive_ev_outcomes:
                total_bet = len(positive_ev_outcomes) * bet_size
                total_expected_profit = sum(
                    calculate_flat_bet(ev_results[o]["ev"], ev_results[o]["prob"], bet_size, threshold)["expected_profit"]
                    for o in positive_ev_outcomes
                )
                embed.add_field(
                    name="📋 Bet Summary",
                    value=(
                        f"**{len(positive_ev_outcomes)}** +EV outcome(s)\n"
                        f"Total stake: `${total_bet:.2f}`\n"
                        f"Total expected profit: `${total_expected_profit:.2f}`"
                    ),
                    inline=True,
                )

            embed.set_footer(text=f"Engine: {SURPLUS_MODEL} via Surplus | EV = (Prob × Odds) − 1 | Flat Bet Mode")

            print(f"[EV CMD] Sending embed...", file=sys.stderr)
            await ctx.send(embed=embed)
            print(f"[EV CMD] Embed sent successfully", file=sys.stderr)

    except Exception as e:
        print(f"[EV CMD] UNHANDLED ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            await ctx.send(f"❌ **Unexpected Error**: {type(e).__name__}: {e}")
        except Exception:
            print(f"[EV CMD] Could not even send error message", file=sys.stderr)


@bot.command(name="evstatus")
async def cmd_ev_status(ctx):
    """Show EV bot configuration status."""
    gid = ctx.guild.id if ctx.guild else ctx.author.id
    threshold = get_ev_threshold(gid)
    bet_size = get_bet_size(gid)
    api_status = "✅ Configured" if SURPLUS_API_KEY else "❌ Not set"

    status_lines = [
        f"**Mode:** Flat Bet (Analysis Only)",
        f"**EV Threshold:** {threshold*100:.1f}%",
        f"**Default Bet Size:** ${bet_size:.2f}",
        f"**AI Model:** {SURPLUS_MODEL}",
        f"**Surplus API:** {api_status}",
    ]

    embed = discord.Embed(
        title="EV Bot Status",
        description="\n".join(status_lines),
        color=discord.Color.green() if SURPLUS_API_KEY else discord.Color.orange(),
    )
    await ctx.send(embed=embed)


# ── Lifecycle ───────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"[EV Bot] ========================================", file=sys.stderr)
    print(f"[EV Bot] Online: {bot.user.name} ({bot.user.id})", file=sys.stderr)
    print(f"[EV Bot] Mode: Flat Bet (Analysis Only)", file=sys.stderr)
    print(f"[EV Bot] Model: {SURPLUS_MODEL}", file=sys.stderr)
    print(f"[EV Bot] EV Threshold: {DEFAULT_EV_THRESHOLD*100:.1f}%", file=sys.stderr)
    print(f"[EV Bot] Default Bet: ${DEFAULT_BET_SIZE_USD:.2f}", file=sys.stderr)
    print(f"[EV Bot] API Key Set: {'Yes' if SURPLUS_API_KEY else 'NO — !ev will fail'}", file=sys.stderr)
    print(f"[EV Bot] ========================================", file=sys.stderr)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    print(f"[EV Bot Error] Command '{ctx.command}': {type(error).__name__}: {error}", file=sys.stderr)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    try:
        await ctx.send(f"❌ **Command Error**: {type(error).__name__}: {error}")
    except Exception:
        print(f"[EV Bot Error] Could not send error message to channel", file=sys.stderr)


async def main():
    if not DISCORD_TOKEN:
        print("FATAL: EV_BOT_TOKEN or DISCORD_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    print(f"[EV Bot] Starting with token: {DISCORD_TOKEN[:8]}...", file=sys.stderr)
    async with bot:
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown():
        print("[EV Bot] Shutting down...", file=sys.stderr)
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
        print("[EV Bot] Stopped cleanly", file=sys.stderr)
