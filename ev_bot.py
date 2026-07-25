"""
ev_bot.py — EV-Based Polymarket Betting Bot
============================================
Architecture:
  1. Gamma API      → fetch market data + clobTokenIds
  2. OpenAI (gpt-5.4) → Pydantic Structured Outputs for fundamental probabilities
  3. CLOB API       → live best-ask prices → decimal odds
  4. EV Math        → deterministic EV = (prob * odds) - 1
  5. py-clob-client → market BUY orders when EV > threshold

Run independently alongside your existing discord_match_bot.py.
"""

import os
import re
import json
import sys
import asyncio
import signal
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
POLYMARKET_PK      = os.getenv("POLYMARKET_PRIVATE_KEY")       # hex private key for py-clob-client
POLYMARKET_FUNDER  = os.getenv("POLYMARKET_FUNDER_ADDRESS")    # your EOA that holds USDC.e on Polygon
DEFAULT_EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.05"))
DEFAULT_BET_SIZE_USD = float(os.getenv("DEFAULT_BET_SIZE", "10.0"))

# ── Pydantic Structured Output Schema ───────────────────────

class MatchProbabilities(BaseModel):
    """Fundamental football probabilities — must sum to 1.0 exactly."""
    home_win: float = Field(..., ge=0.0, le=1.0, description="Probability home team wins")
    draw: float = Field(..., ge=0.0, le=1.0, description="Probability of a draw")
    away_win: float = Field(..., ge=0.0, le=1.0, description="Probability away team wins")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    match_name: str = Field(..., description="Full match name")
    key_factors: str = Field(..., description="1-sentence summary of key fundamental factors")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in these probabilities")

# JSON Schema for OpenAI structured output (compatible with any OpenAI-compatible API)
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

# ── System Prompt ───────────────────────────────────────────

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
    """Fetch event details from Polymarket Gamma API."""
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
    except Exception as e:
        print(f"[Gamma] Event lookup failed: {e}", file=sys.stderr)
    return None


def extract_markets_from_event(event: dict) -> list[dict]:
    """Extract home/draw/away markets with their clobTokenIds from a Gamma event."""
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

        # Classify market type
        if "draw" in question or "tie" in question:
            market_type = "draw"
        elif "win" in question:
            # Determine home or away
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
            else:
                # Fallback: first team mentioned = home
                if home_name and any(w in question for w in home_name.split()):
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
    """Extract event slug from a Polymarket URL."""
    m = re.search(
        r'https?://polymarket\.com/(?:event|sports/[a-z0-9]+)/([a-z0-9][a-z0-9\-]+[a-z0-9])',
        url, re.IGNORECASE
    )
    return m.group(1) if m else None


# ── CLOB API ────────────────────────────────────────────────

def fetch_clob_best_ask(token_id: str) -> Optional[float]:
    """Fetch the best ask price for a token from Polymarket CLOB.
    Returns the share price (0.0 to 1.0) or None."""
    url = f"https://clob.polymarket.com/price?token_id={token_id}&side=buy"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                price_str = data.get("price", "0")
                return float(price_str)
    except Exception as e:
        print(f"[CLOB] Price fetch failed for token {token_id}: {e}", file=sys.stderr)
    return None


def share_price_to_decimal_odds(price: float) -> float:
    """Convert CLOB share price (0-1) to decimal odds.
    Decimal Odds = 1 / Share Price"""
    if price <= 0:
        return float("inf")
    return 1.0 / price


# ── OpenAI Probability Engine ───────────────────────────────

def get_openai_client() -> AsyncOpenAI:
    if not SURPLUS_API_KEY:
        raise RuntimeError("SURPLUS_API_KEY not set")
    return AsyncOpenAI(api_key=SURPLUS_API_KEY, base_url=SURPLUS_BASE_URL)


async def fetch_fundamental_probabilities(
    match_title: str,
    home_team: str,
    away_team: str,
    league: str = "Unknown",
    match_date: str = "Unknown",
) -> MatchProbabilities:
    """Call OpenAI with Structured Outputs to get fundamental probabilities."""
    client = get_openai_client()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        match_title=match_title,
        home_team=home_team,
        away_team=away_team,
        league=league,
        match_date=match_date,
    )

    # Try structured output first, fall back to raw JSON parsing
    try:
        response = await client.chat.completions.create(
            model=SURPLUS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format=MATCH_PROB_SCHEMA,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("empty response")

        data = json.loads(raw)

        # Validate sum
        total = data["home_win"] + data["draw"] + data["away_win"]
        if abs(total - 1.0) > 0.001:
            # Normalize
            data["home_win"] /= total
            data["draw"] /= total
            data["away_win"] /= total

        return MatchProbabilities(**data)

    except Exception as e:
        print(f"[OpenAI] Structured output failed ({e}), falling back to raw JSON parsing", file=sys.stderr)
        # Fallback: raw completion + manual parsing
        response = await client.chat.completions.create(
            model=SURPLUS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content or ""
        return _parse_raw_probabilities(raw)


def _parse_raw_probabilities(raw: str) -> MatchProbabilities:
    """Fallback parser for when structured outputs aren't supported."""
    # Extract JSON from response
    json_str = raw
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL | re.IGNORECASE)
    if m:
        json_str = m.group(1).strip()

    # Find JSON object
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start >= 0 and end > start:
        json_str = json_str[start:end+1]

    data = json.loads(json_str)

    # Normalize
    total = data.get("home_win", 0) + data.get("draw", 0) + data.get("away_win", 0)
    if total > 0 and abs(total - 1.0) > 0.001:
        data["home_win"] = data.get("home_win", 0) / total
        data["draw"] = data.get("draw", 0) / total
        data["away_win"] = data.get("away_win", 0) / total

    return MatchProbabilities(
        home_win=data.get("home_win", 0.33),
        draw=data.get("draw", 0.34),
        away_win=data.get("away_win", 0.33),
        home_team=data.get("home_team", "Home"),
        away_team=data.get("away_team", "Away"),
        match_name=data.get("match_name", "Unknown"),
        key_factors=data.get("key_factors", "No factors provided"),
        confidence=data.get("confidence", 0.5),
    )


# ── EV Calculation ─────────────────────────────────────────

def calculate_ev(
    probs: MatchProbabilities,
    clob_prices: dict[str, Optional[float]],
) -> dict:
    """Calculate Expected Value for each outcome.
    EV = (AI Probability * Decimal Odds) - 1

    Returns dict with EV values and decision flags.
    """
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


# ── py-clob-client Execution ────────────────────────────────

def get_clob_client():
    """Initialize py-clob-client. Returns None if not configured."""
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        print("[CLOB Client] py-clob-client not installed. Run: pip install py-clob-client", file=sys.stderr)
        return None

    if not POLYMARKET_PK:
        print("[CLOB Client] POLYMARKET_PRIVATE_KEY not set", file=sys.stderr)
        return None

    funder = POLYMARKET_FUNDER or None

    try:
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=POLYMARKET_PK,
            chain_id=137,       # Polygon
            signature_type=1,   # EOA
            funder=funder,
        )
        # Create API credentials if needed
        try:
            creds = client.create_api_key()
            print(f"[CLOB Client] API credentials created", file=sys.stderr)
        except Exception:
            # May already exist
            pass
        return client
    except Exception as e:
        print(f"[CLOB Client] Failed to initialize: {e}", file=sys.stderr)
        return None


def place_market_buy_order(token_id: str, bet_amount_usd: float) -> Optional[dict]:
    """Place a market BUY order on Polymarket CLOB.
    
    Args:
        token_id: The clobTokenId for the outcome
        bet_amount_usd: Dollar amount to bet (used to calculate share count)
    
    Returns order response dict or None on failure.
    """
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY
    except ImportError:
        return {"error": "py-clob-client not installed"}

    client = get_clob_client()
    if client is None:
        return {"error": "CLOB client not configured — check POLYMARKET_PRIVATE_KEY"}

    # Get current best ask price
    price = fetch_clob_best_ask(token_id)
    if price is None or price <= 0:
        return {"error": f"Could not fetch price for token {token_id}"}

    # Calculate shares: shares = bet_amount / price_per_share
    shares = bet_amount_usd / price

    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=BUY,
        )
        signed_order = client.create_order(order_args)
        response = client.post_order(signed_order)
        return {
            "success": True,
            "token_id": token_id,
            "price": price,
            "shares": shares,
            "bet_amount": bet_amount_usd,
            "order_id": response.get("orderID") or response.get("id"),
            "response": response,
        }
    except Exception as e:
        return {"error": str(e), "token_id": token_id}


# ── Discord Bot ─────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Per-guild EV threshold
ev_thresholds = {}


def get_ev_threshold(guild_id: int) -> float:
    return ev_thresholds.get(guild_id, DEFAULT_EV_THRESHOLD)


def set_ev_threshold(guild_id: int, threshold: float):
    ev_thresholds[guild_id] = threshold


@bot.command(name="ev")
async def cmd_ev(ctx, *, args: str = ""):
    """Analyze a Polymarket football event and calculate EV.
    
    Usage:
      !ev <polymarket_url>
      !ev <polymarket_url> --bet <amount>
      !ev <polymarket_url> --auto
    
    --bet <amount>: Place bets on all +EV outcomes with specified USD amount each
    --auto: Auto-bet default amount ($10) on all +EV outcomes
    """
    if not args:
        await ctx.send(
            "**EV Bot Usage:**\n"
            "`!ev <polymarket_url>` — Analyze match and show EV\n"
            "`!ev <polymarket_url> --bet <amount>` — Bet $X on each +EV outcome\n"
            "`!ev <polymarket_url> --auto` — Auto-bet $10 on each +EV outcome\n"
            "`!ev threshold <0.05>` — Set EV threshold"
        )
        return

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

    # ── Parse args ──
    bet_mode = False
    bet_amount = DEFAULT_BET_SIZE_USD
    auto_mode = False

    if "--auto" in args:
        auto_mode = True
        bet_mode = True
        args = args.replace("--auto", "").strip()

    bet_match = re.search(r'--bet\s+(\d+(?:\.\d+)?)', args)
    if bet_match:
        bet_amount = float(bet_match.group(1))
        bet_mode = True
        args = args.replace(bet_match.group(0), "").strip()

    url = args.strip()

    # ── Extract slug ──
    slug = parse_polymarket_url(url)
    if not slug:
        await ctx.send("❌ Could not extract event slug from URL. Use a Polymarket event URL.")
        return

    async with ctx.typing():
        # ── Step 1: Fetch Gamma event ──
        event = await asyncio.to_thread(fetch_event_from_gamma, slug)
        if not event:
            await ctx.send(f"❌ Event not found for slug `{slug}`")
            return

        title = event.get("title", slug)
        teams = event.get("teams", [])
        home_team = teams[0].get("name", "Home") if len(teams) >= 1 else "Home"
        away_team = teams[1].get("name", "Away") if len(teams) >= 2 else "Away"

        # Extract league/series info
        series = event.get("series", [])
        league = "Unknown"
        if isinstance(series, list) and len(series) > 0:
            league = series[0].get("title", "Unknown")
        elif isinstance(event.get("sport"), dict):
            league = event["sport"].get("sport", "Unknown").upper()

        # Match date
        match_date = event.get("startDate") or event.get("scheduledStart") or "Unknown"

        # ── Step 2: Extract markets ──
        markets = extract_markets_from_event(event)
        if len(markets) < 2:
            await ctx.send(f"❌ Could not find enough markets (home/draw/away) for this event. Found: {len(markets)}")
            return

        # Build market lookup
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
                    "price": price,
                    "token_id": m["token_id"],
                    "question": m.get("question", ""),
                }
            else:
                clob_prices[outcome_type] = {"price": None, "token_id": None, "question": ""}

        # ── Step 4: AI Probability Engine ──
        try:
            probs = await fetch_fundamental_probabilities(
                match_title=title,
                home_team=home_team,
                away_team=away_team,
                league=league,
                match_date=str(match_date),
            )
        except Exception as e:
            await ctx.send(f"❌ **AI Engine Error**: {e}")
            return

        # ── Step 5: EV Calculation ──
        ev_results = calculate_ev(probs, clob_prices)
        gid = ctx.guild.id if ctx.guild else ctx.author.id
        threshold = get_ev_threshold(gid)

        # ── Build Discord Embed ──
        embed = discord.Embed(
            title=f"📊 EV Analysis: {probs.match_name}",
            description=f"**League:** {league} | **Date:** {match_date}\n**Slug:** `{slug}`",
            color=discord.Color.blue(),
        )

        # Probabilities
        prob_lines = [
            f"🏠 **{probs.home_team}**: {probs.home_win*100:.1f}%",
            f"🤝 **Draw**: {probs.draw*100:.1f}%",
            f"🚶 **{probs.away_team}**: {probs.away_win*100:.1f}%",
        ]
        embed.add_field(name="AI Fundamental Probabilities", value="\n".join(prob_lines), inline=False)

        # CLOB Prices + EV
        ev_lines = []
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
                ev_lines.append(f"✅ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV **{ev_str}** ← +EV")
                positive_ev_outcomes.append(outcome)
            elif r["ev"] is not None:
                ev_lines.append(f"❌ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV {ev_str}")
            else:
                ev_lines.append(f"⚪ {emoji} **{label}**: Price {price_str} | No EV data")

        embed.add_field(
            name=f"CLOB Prices & EV (threshold: {threshold*100:.1f}%)",
            value="\n".join(ev_lines),
            inline=False,
        )

        # Key factors
        embed.add_field(name="Key Factors", value=probs.key_factors, inline=False)
        embed.add_field(name="AI Confidence", value=f"{probs.confidence*100:.0f}%", inline=True)

        embed.set_footer(text=f"Engine: {SURPLUS_MODEL} via Surplus | EV = (Prob × Odds) − 1")

        await ctx.send(embed=embed)

        # ── Step 6: Execution (if --bet or --auto) ──
        if bet_mode and positive_ev_outcomes:
            if not POLYMARKET_PK:
                await ctx.send(
                    "⚠️ **Cannot place bets**: `POLYMARKET_PRIVATE_KEY` not set.\n"
                    "Set it in your environment variables to enable execution."
                )
                return

            await ctx.send(f"⚡ **Placing bets** (${bet_amount:.2f} each on +EV outcomes)...")

            for outcome in positive_ev_outcomes:
                m = market_lookup.get(outcome)
                if not m:
                    continue

                label = {"home": probs.home_team, "draw": "Draw", "away": probs.away_team}[outcome]
                ev_val = ev_results[outcome]["ev"]

                result = await asyncio.to_thread(place_market_buy_order, m["token_id"], bet_amount)

                if result and result.get("success"):
                    await ctx.send(
                        f"✅ **Bet placed: {label}**\n"
                        f"• Amount: `${bet_amount:.2f}` → `{result['shares']:.2f}` shares @ `{result['price']*100:.1f}¢`\n"
                        f"• EV: `{ev_val*100:+.1f}%`\n"
                        f"• Order ID: `{result['order_id']}`"
                    )
                else:
                    error_msg = result.get("error", "Unknown error") if result else "No result"
                    await ctx.send(f"❌ **Bet failed: {label}** — {error_msg}")

        elif bet_mode and not positive_ev_outcomes:
            await ctx.send("ℹ️ No outcomes exceed the EV threshold — no bets placed.")

        elif not bet_mode and positive_ev_outcomes:
            outcomes_str = ", ".join(
                {"home": probs.home_team, "draw": "Draw", "away": probs.away_team}[o]
                for o in positive_ev_outcomes
            )
            await ctx.send(
                f"💡 **+EV opportunities detected**: {outcomes_str}\n"
                f"Bet with: `!ev {url} --bet {DEFAULT_BET_SIZE_USD}` or `!ev {url} --auto`"
            )


@bot.command(name="evstatus")
async def cmd_ev_status(ctx):
    """Show EV bot configuration status."""
    gid = ctx.guild.id if ctx.guild else ctx.author.id
    threshold = get_ev_threshold(gid)

    pk_status = "✅ Configured" if POLYMARKET_PK else "❌ Not set"
    funder_status = POLYMARKET_FUNDER or "Not set (using derived address)"
    api_status = "✅ Configured" if SURPLUS_API_KEY else "❌ Not set"

    clob_available = False
    try:
        import py_clob_client
        clob_available = True
    except ImportError:
        pass

    status_lines = [
        f"**EV Threshold:** {threshold*100:.1f}%",
        f"**AI Model:** {SURPLUS_MODEL}",
        f"**Surplus API:** {api_status}",
        f"**Private Key:** {pk_status}",
        f"**Funder Address:** {funder_status}",
        f"**py-clob-client:** {'✅ Installed' if clob_available else '❌ Not installed'}",
        f"**Default Bet Size:** ${DEFAULT_BET_SIZE_USD:.2f}",
    ]

    embed = discord.Embed(
        title="EV Bot Status",
        description="\n".join(status_lines),
        color=discord.Color.green() if POLYMARKET_PK and SURPLUS_API_KEY else discord.Color.orange(),
    )
    await ctx.send(embed=embed)


# ── Lifecycle ───────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"[EV Bot] Online: {bot.user.name} ({bot.user.id})")
    print(f"[EV Bot] Model: {SURPLUS_MODEL}")
    print(f"[EV Bot] EV Threshold: {DEFAULT_EV_THRESHOLD*100:.1f}%")
    print(f"[EV Bot] Default Bet: ${DEFAULT_BET_SIZE_USD:.2f}")
    print(f"[EV Bot] PK Configured: {'Yes' if POLYMARKET_PK else 'No'}")
    print(f"[EV Bot] py-clob-client: ", end="")
    try:
        import py_clob_client
        print("Installed")
    except ImportError:
        print("Not installed — run: pip install py-clob-client")


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    print(f"[EV Bot Error] {ctx.command}: {error}", file=sys.stderr)
    await ctx.send(f"❌ **Error**: {error}")


async def main():
    if not DISCORD_TOKEN:
        print("FATAL: EV_BOT_TOKEN or DISCORD_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)
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
