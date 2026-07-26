"""
ev_bot.py — EV-Based Polymarket Betting Bot (Kelly Criterion Mode)
======================================================================
Architecture:
  1. Gamma API      → fetch market data + clobTokenIds
  2. OpenAI (gpt-5.4) → compressed prompt, bivariate Poisson, outputs final probs
  3. CLOB API       → live best-ask prices → decimal odds
  4. EV Math        → deterministic EV = (prob * odds) - 1
  5. Kelly Staking  → f* = EV / (odds - 1), scaled by kelly fraction

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
DEFAULT_KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "1000.0"))

# ── Pydantic Output Schema ──────────────────────────────────

class MatchProbabilities(BaseModel):
    home_win: float = Field(..., ge=0.0, le=1.0)
    draw: float = Field(..., ge=0.0, le=1.0)
    away_win: float = Field(..., ge=0.0, le=1.0)
    home_team: str = Field(...)
    away_team: str = Field(...)
    match_name: str = Field(...)
    justification: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)

# ── Compressed Prompts ──────────────────────────────────────

SYSTEM_PROMPT = (
    "Role: Deterministic Quant Sports Analyst.\n"
    "Task: Calculate true win/draw/loss probability for football.\n"
    "Rules:\n"
    "1. Use purely fundamental data (form, xG, H2H, injuries, venue).\n"
    "2. IGNORE ALL MARKET ODDS AND CONSENSUS.\n"
    "3. Run bivariate Poisson distribution internally based on derived expected goals (xG). "
    "Home venue = +0.20 xG, Away = -0.20 xG.\n"
    "4. OUTPUT FORMAT STRICT: Output ONLY the final % for Home/Draw/Away, followed by a "
    "maximum 2-sentence mathematical justification. No conversational filler. No step-by-step math.\n\n"
    "Output exactly this JSON and nothing else:\n"
    '{"home_win":0.XX,"draw":0.XX,"away_win":0.XX,"justification":"<2 sentences max>"}\n'
    "Probabilities MUST sum to 1.0."
)

USER_PROMPT_TEMPLATE = (
    "[{league}] | Home: {home_team} | Away: {away_team}\n"
    "H_Data: {home_data}\n"
    "A_Data: {away_data}\n"
    "H2H_Venue: {h2h}\n"
    "Output probabilities."
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


# ── Data Compression ────────────────────────────────────────

def compress_team_data(event: dict, team_idx: int) -> str:
    """
    Compress team stats into dense format: [W-D-L], [GF_Avg], [GA_Avg], [Injuries]
    Falls back to placeholder if Gamma doesn't provide stats.
    """
    teams = event.get("teams", [])
    if team_idx >= len(teams):
        return "N/A,N/A,N/A,N/A"

    team = teams[team_idx]
    name = team.get("name", "Unknown")

    # Try to extract from Gamma's nested data
    # Gamma sometimes provides stats in team metadata
    stats = team.get("stats", {}) or {}

    wins = stats.get("wins", "?")
    draws = stats.get("draws", "?")
    losses = stats.get("losses", "?")
    gf = stats.get("goalsFor", stats.get("gf", "?"))
    ga = stats.get("goalsAgainst", stats.get("ga", "?"))

    # Check for injuries in event metadata
    injuries = event.get("injuries", {}) or {}
    team_injuries = injuries.get(name, injuries.get(name.lower(), "None"))

    # If no stats at all, use a minimal placeholder — the LLM will still work
    if wins == "?" and draws == "?" and losses == "?":
        return f"[?-?-?], [GF:{gf}], [GA:{ga}], [{team_injuries}]"

    return f"[{wins}-{draws}-{losses}], [GF:{gf}], [GA:{ga}], [{team_injuries}]"


def compress_h2h(event: dict) -> str:
    """Compress H2H data from event metadata."""
    h2h = event.get("headToHead", event.get("h2h", {})) or {}
    if not h2h:
        return "No data"

    # Try common Gamma H2H formats
    last_score = h2h.get("lastScore", h2h.get("last_score", ""))
    if last_score:
        return str(last_score)

    last_matches = h2h.get("lastMatches", h2h.get("last_matches", []))
    if last_matches and len(last_matches) > 0:
        return str(last_matches[0])

    return "No data"


# ── OpenAI Probability Engine ───────────────────────────────

def get_openai_client() -> AsyncOpenAI:
    if not SURPLUS_API_KEY:
        raise RuntimeError("SURPLUS_API_KEY not set")
    return AsyncOpenAI(api_key=SURPLUS_API_KEY, base_url=SURPLUS_BASE_URL)


def _extract_json_from_text(text: str) -> dict:
    """Robust JSON extraction from LLM output."""
    cleaned = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end+1]
    return json.loads(cleaned)


async def fetch_fundamental_probabilities(
    match_title: str, home_team: str, away_team: str,
    league: str = "Unknown", match_date: str = "Unknown",
    event: Optional[dict] = None,
) -> MatchProbabilities:
    """
    LLM calculates final probabilities directly using bivariate Poisson.
    Python only handles EV and Kelly — no probability math here.
    """
    client = get_openai_client()

    # Compress team data
    if event:
        home_data = compress_team_data(event, 0)
        away_data = compress_team_data(event, 1)
        h2h = compress_h2h(event)
    else:
        home_data = "N/A,N/A,N/A,N/A"
        away_data = "N/A,N/A,N/A,N/A"
        h2h = "No data"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        league=league,
        home_team=home_team,
        away_team=away_team,
        home_data=home_data,
        away_data=away_data,
        h2h=h2h,
    )

    print(f"[OpenAI] Compressed prompt ({len(user_prompt)} chars): {user_prompt}", file=sys.stderr)

    # Try json_object mode first
    try:
        response = await client.chat.completions.create(
            model=SURPLUS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("empty response from LLM")
        print(f"[OpenAI] Raw response ({len(raw)} chars): {raw}", file=sys.stderr)
        data = _extract_json_from_text(raw)
    except Exception as e:
        print(f"[OpenAI] json_object mode failed ({e}), trying raw completion", file=sys.stderr)
        response = await client.chat.completions.create(
            model=SURPLUS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content or ""
        print(f"[OpenAI] Raw fallback response ({len(raw)} chars): {raw}", file=sys.stderr)
        data = _extract_json_from_text(raw)

    # LLM outputs final probabilities — Python just validates
    home = float(data.get("home_win", 0.33))
    draw = float(data.get("draw", 0.34))
    away = float(data.get("away_win", 0.33))
    total = home + draw + away

    if total > 0 and abs(total - 1.0) > 0.001:
        home /= total
        draw /= total
        away /= total

    return MatchProbabilities(
        home_win=home,
        draw=draw,
        away_win=away,
        home_team=home_team,
        away_team=away_team,
        match_name=match_title,
        justification=str(data.get("justification", "No justification provided")),
        confidence=float(data.get("confidence", 0.5)),
    )


# ── EV Calculation (Python only — no probability math) ─────

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


# ── Kelly Criterion Staking (Python only) ───────────────────

def calculate_kelly_bet(
    ev: float,
    prob: float,
    odds: float,
    bankroll: float,
    kelly_fraction: float,
    threshold: float,
) -> Optional[dict]:
    """
    Kelly Criterion: f* = EV / (odds - 1)
    Scaled by kelly_fraction, capped at 25% of bankroll.
    """
    if ev < threshold or odds <= 1.0:
        return None

    b = odds - 1.0
    if b <= 0:
        return None

    full_kelly_pct = ev / b
    if full_kelly_pct <= 0:
        return None

    scaled_kelly_pct = full_kelly_pct * kelly_fraction
    scaled_kelly_pct = min(scaled_kelly_pct, 0.25)

    stake = bankroll * scaled_kelly_pct
    expected_profit = stake * ev

    return {
        "full_kelly_pct": full_kelly_pct * 100,
        "kelly_fraction": kelly_fraction,
        "scaled_kelly_pct": scaled_kelly_pct * 100,
        "stake": stake,
        "expected_profit": expected_profit,
        "expected_roi_pct": ev * 100,
    }


# ── Discord Bot ─────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

ev_thresholds = {}
kelly_fractions = {}
bankrolls = {}

def get_ev_threshold(guild_id: int) -> float:
    return ev_thresholds.get(guild_id, DEFAULT_EV_THRESHOLD)

def set_ev_threshold(guild_id: int, threshold: float):
    ev_thresholds[guild_id] = threshold

def get_kelly_fraction(guild_id: int) -> float:
    return kelly_fractions.get(guild_id, DEFAULT_KELLY_FRACTION)

def set_kelly_fraction(guild_id: int, fraction: float):
    kelly_fractions[guild_id] = fraction

def get_bankroll(guild_id: int) -> float:
    return bankrolls.get(guild_id, DEFAULT_BANKROLL)

def set_bankroll(guild_id: int, bankroll: float):
    bankrolls[guild_id] = bankroll


@bot.command(name="ev")
async def cmd_ev(ctx, *, args: str = ""):
    """Analyze a Polymarket football event and calculate EV with Kelly staking."""
    print(f"[EV CMD] Received !ev from {ctx.author} with args: '{args}'", file=sys.stderr)

    if not args:
        help_text = (
            "**EV Bot — Kelly Criterion Mode**\n"
            "`!ev <polymarket_url>` — Analyze match, show EV + Kelly stakes\n"
            "`!ev threshold <0.05>` — Set EV threshold (e.g. 0.05 = 5%)\n"
            "`!ev kelly <0.5>` — Set Kelly fraction (0.5 = half-Kelly, 0.25 = quarter-Kelly)\n"
            "`!ev bankroll <1000>` — Set bankroll size in USD\n"
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

        # ── Handle Kelly fraction setting ──
        if args.lower().startswith("kelly"):
            try:
                parts = args.split()
                if len(parts) >= 2:
                    new_kelly = float(parts[1])
                    if new_kelly <= 0 or new_kelly > 1.0:
                        await ctx.send("❌ Kelly fraction must be between 0.01 and 1.0")
                        return
                    gid = ctx.guild.id if ctx.guild else ctx.author.id
                    set_kelly_fraction(gid, new_kelly)
                    label = "full-Kelly" if new_kelly == 1.0 else f"{new_kelly*100:.0f}%-Kelly"
                    await ctx.send(f"✅ Kelly fraction set to **{new_kelly}** ({label})")
                    return
            except ValueError:
                await ctx.send("❌ Invalid Kelly fraction. Use: `!ev kelly 0.5`")
                return

        # ── Handle bankroll setting ──
        if args.lower().startswith("bankroll"):
            try:
                parts = args.split()
                if len(parts) >= 2:
                    new_br = float(parts[1])
                    if new_br <= 0:
                        await ctx.send("❌ Bankroll must be positive")
                        return
                    gid = ctx.guild.id if ctx.guild else ctx.author.id
                    set_bankroll(gid, new_br)
                    await ctx.send(f"✅ Bankroll set to **${new_br:,.2f}**")
                    return
            except ValueError:
                await ctx.send("❌ Invalid bankroll. Use: `!ev bankroll 1000`")
                return

        # ── Parse URL ──
        url = args.strip()
        print(f"[EV CMD] URL to analyze: '{url}'", file=sys.stderr)

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

            # ── Step 4: AI Probability Engine (LLM does ALL probability math) ──
            print(f"[EV CMD] Calling AI for: {title}", file=sys.stderr)
            try:
                probs = await fetch_fundamental_probabilities(
                    match_title=title, home_team=home_team, away_team=away_team,
                    league=league, match_date=str(match_date), event=event,
                )
            except Exception as e:
                await ctx.send(f"❌ **AI Engine Error**: {type(e).__name__}: {e}\nCheck SURPLUS_API_KEY and model `{SURPLUS_MODEL}`.")
                return

            print(f"[EV CMD] AI probs: home={probs.home_win:.3f}, draw={probs.draw:.3f}, away={probs.away_win:.3f}", file=sys.stderr)

            # ── Step 5: EV Calculation (Python only) ──
            ev_results = calculate_ev(probs, clob_prices)
            gid = ctx.guild.id if ctx.guild else ctx.author.id
            threshold = get_ev_threshold(gid)
            kelly_frac = get_kelly_fraction(gid)
            bankroll = get_bankroll(gid)

            # ── Build Discord Embed ──
            kelly_label = "Full-Kelly" if kelly_frac == 1.0 else f"{kelly_frac*100:.0f}%-Kelly"
            embed = discord.Embed(
                title=f"📊 EV Analysis: {probs.match_name}",
                description=(
                    f"**League:** {league} | **Date:** {match_date}\n"
                    f"**Slug:** `{slug}`\n"
                    f"**Staking:** {kelly_label} | **Bankroll:** ${bankroll:,.2f} | **Threshold:** {threshold*100:.1f}%"
                ),
                color=discord.Color.blue(),
            )

            # Probabilities (LLM output directly)
            prob_lines = [
                f"🏠 **{probs.home_team}**: {probs.home_win*100:.1f}%",
                f"🤝 **Draw**: {probs.draw*100:.1f}%",
                f"🚶 **{probs.away_team}**: {probs.away_win*100:.1f}%",
            ]
            embed.add_field(name="AI Fundamental Probabilities (Bivariate Poisson)", value="\n".join(prob_lines), inline=False)

            # Justification
            embed.add_field(name="📐 Mathematical Justification", value=probs.justification, inline=False)

            # CLOB Prices + EV + Kelly Stakes
            ev_lines = []
            kelly_lines = []
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

                if r["ev"] is not None and r["ev"] >= threshold and r["odds"] is not None:
                    ev_lines.append(f"✅ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV **{ev_str}**")
                    positive_ev_outcomes.append(outcome)

                    kelly = calculate_kelly_bet(r["ev"], r["prob"], r["odds"], bankroll, kelly_frac, threshold)
                    if kelly:
                        kelly_lines.append(
                            f"{emoji} **{label}**: Stake **${kelly['stake']:,.2f}** "
                            f"({kelly['scaled_kelly_pct']:.1f}% of bankroll)\n"
                            f"　↳ Full Kelly: {kelly['full_kelly_pct']:.1f}% | "
                            f"Expected profit: **${kelly['expected_profit']:,.2f}** "
                            f"({kelly['expected_roi_pct']:+.1f}% ROI)"
                        )
                elif r["ev"] is not None:
                    ev_lines.append(f"❌ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV {ev_str}")
                else:
                    ev_lines.append(f"⚪ {emoji} **{label}**: Price {price_str} | No EV data")

            embed.add_field(
                name=f"CLOB Prices & EV",
                value="\n".join(ev_lines) if ev_lines else "No EV data available",
                inline=False,
            )

            if kelly_lines:
                embed.add_field(
                    name=f"💰 Kelly Stakes ({kelly_label}, Bankroll: ${bankroll:,.2f})",
                    value="\n".join(kelly_lines),
                    inline=False,
                )

            embed.add_field(name="AI Confidence", value=f"{probs.confidence*100:.0f}%", inline=True)

            # Summary
            if positive_ev_outcomes:
                total_stake = sum(
                    calculate_kelly_bet(
                        ev_results[o]["ev"], ev_results[o]["prob"],
                        ev_results[o]["odds"], bankroll, kelly_frac, threshold
                    )["stake"]
                    for o in positive_ev_outcomes
                )
                total_expected_profit = sum(
                    calculate_kelly_bet(
                        ev_results[o]["ev"], ev_results[o]["prob"],
                        ev_results[o]["odds"], bankroll, kelly_frac, threshold
                    )["expected_profit"]
                    for o in positive_ev_outcomes
                )
                total_kelly_pct = (total_stake / bankroll) * 100 if bankroll > 0 else 0
                embed.add_field(
                    name="📋 Bet Summary",
                    value=(
                        f"**{len(positive_ev_outcomes)}** +EV outcome(s)\n"
                        f"Total stake: **${total_stake:,.2f}** ({total_kelly_pct:.1f}% of bankroll)\n"
                        f"Total expected profit: **${total_expected_profit:,.2f}**"
                    ),
                    inline=True,
                )
            else:
                embed.add_field(
                    name="📋 Bet Summary",
                    value="No +EV outcomes above threshold — no bets recommended.",
                    inline=True,
                )

            embed.set_footer(text=f"Engine: {SURPLUS_MODEL} via Surplus | Kelly: f* = EV/(odds−1) × {kelly_frac} | Max 25%/bet")

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
    kelly_frac = get_kelly_fraction(gid)
    bankroll = get_bankroll(gid)
    api_status = "✅ Configured" if SURPLUS_API_KEY else "❌ Not set"

    kelly_label = "Full-Kelly" if kelly_frac == 1.0 else f"{kelly_frac*100:.0f}%-Kelly"

    status_lines = [
        f"**Mode:** Kelly Criterion Staking",
        f"**EV Threshold:** {threshold*100:.1f}%",
        f"**Kelly Fraction:** {kelly_frac} ({kelly_label})",
        f"**Bankroll:** ${bankroll:,.2f}",
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
    print(f"[EV Bot] Mode: Kelly Criterion Staking", file=sys.stderr)
    print(f"[EV Bot] Model: {SURPLUS_MODEL}", file=sys.stderr)
    print(f"[EV Bot] EV Threshold: {DEFAULT_EV_THRESHOLD*100:.1f}%", file=sys.stderr)
    print(f"[EV Bot] Kelly Fraction: {DEFAULT_KELLY_FRACTION}", file=sys.stderr)
    print(f"[EV Bot] Bankroll: ${DEFAULT_BANKROLL:,.2f}", file=sys.stderr)
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
