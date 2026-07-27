"""
mlb_bot.py — MLB Sabermetric EV Betting Bot (Kelly Criterion Mode)
=====================================================================
Architecture:
  1. Gamma API         → fetch MLB market data + clobTokenIds
  2. MLB Stats API     → scrape probable pitchers for today's games
  3. OpenAI (gpt-5.4)  → isolated sabermetric variables (NOT probabilities)
  4. Python Backend    → deterministic probability calculation from sabermetrics
  5. CLOB API          → live best-ask prices → decimal odds
  6. EV Math           → EV = (prob * odds) - 1
  7. Kelly Staking     → f* = EV / (odds - 1), scaled by kelly fraction

Strict separation: LLM outputs raw structural data only.
ALL final probability calculations, risk management, and execution
decisions are handled exclusively by Python machine code.
"""

import os, re, json, sys, asyncio, signal, traceback, urllib.request, urllib.parse, math
from typing import Optional

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

import discord
from discord.ext import commands

# ── env config ──────────────────────────────────────────────
DISCORD_TOKEN      = os.getenv("MLB_BOT_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")
SURPLUS_API_KEY    = os.getenv("SURPLUS_API_KEY")
SURPLUS_BASE_URL   = os.getenv("SURPLUS_API_URL", "https://api.surplusintelligence.ai/min30/v1")
SURPLUS_MODEL      = os.getenv("SURPLUS_MODEL", "gpt-5.4")
DEFAULT_EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.05"))
DEFAULT_KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "1000.0"))

# ── Helpers ─────────────────────────────────────────────────

def _strip_none(d: dict) -> dict:
    """Remove keys with None values so Pydantic defaults apply."""
    return {k: v for k, v in d.items() if v is not None}

def _extract_date_ymd(raw) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp, date string, or 'YYYY-MM-DD HH:MM:SS+00' format."""
    if not raw or raw == "Unknown":
        return "Unknown"
    s = str(raw).strip()
    m = re.match(r'(\d{4}-\d{2}-\d{2})', s)
    return m.group(1) if m else s[:10]

# ── Pydantic Output Schema ──────────────────────────────────

class PitcherData(BaseModel):
    name: str = Field(default="Unknown")
    handedness: str = Field(default="R")
    xfip: Optional[float] = Field(default=None)
    siera: Optional[float] = Field(default=None)
    k_bb_pct: Optional[float] = Field(default=None)
    vs_lhb: Optional[float] = Field(default=None)
    vs_rhb: Optional[float] = Field(default=None)

class OffenseData(BaseModel):
    wrc_plus: Optional[float] = Field(default=None)
    iso: Optional[float] = Field(default=None)
    ops: Optional[float] = Field(default=None)

class BullpenData(BaseModel):
    rested_key_relievers: Optional[int] = Field(default=None)
    aggregate_xfip: Optional[float] = Field(default=None)
    fatigue_flag: Optional[str] = Field(default=None)

class VenueData(BaseModel):
    park_factor: Optional[float] = Field(default=None)
    hr_factor: Optional[float] = Field(default=None)
    wind_speed_mph: Optional[float] = Field(default=None)
    wind_direction: Optional[str] = Field(default=None)
    temp_f: Optional[float] = Field(default=None)
    stadium_name: str = Field(default="Unknown")

class SabermetricOutput(BaseModel):
    away_team: str = Field(...)
    home_team: str = Field(...)
    match_name: str = Field(...)
    away_pitcher: PitcherData = Field(default_factory=PitcherData)
    home_pitcher: PitcherData = Field(default_factory=PitcherData)
    away_offense: OffenseData = Field(default_factory=OffenseData)
    home_offense: OffenseData = Field(default_factory=OffenseData)
    away_bullpen: BullpenData = Field(default_factory=BullpenData)
    home_bullpen: BullpenData = Field(default_factory=BullpenData)
    venue: VenueData = Field(default_factory=VenueData)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

# ── Prompts ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "**Role & Objective**\n"
    "You are the sabermetric data extraction module of an automated MLB prediction framework. "
    "Your primary role is to fetch, parse, and verify real-time game conditions, starting pitchers, "
    "and stadium factors for a given matchup.\n\n"
    "**CRITICAL DATA FETCHING RULES**\n\n"
    "1. **VENUE INFERENCE (Never Output 'Unknown'):**\n"
    "   - The venue is ALWAYS the home team's official ballpark unless explicitly designated "
    "as an international/neutral site game.\n"
    "   - You MUST automatically fill the venue based on the home team (e.g., Phillies = "
    "Citizens Bank Park, Giants = Oracle Park, Yankees = Yankee Stadium, Dodgers = "
    "Dodger Stadium).\n\n"
    "2. **STARTING PITCHERS:** The confirmed starting pitchers are provided in the user prompt. "
    "Use them directly — do NOT search for pitchers. If a pitcher is listed as 'TBD', "
    "that means no official announcement exists yet.\n\n"
    "3. **EXCLUSIONS & HANDOFF:**\n"
    "   - Output ONLY raw fundamental variables (pitcher xFIP/SIERA, venue factors, "
    "bullpen fatigue, offense wRC+).\n"
    "   - NEVER calculate a final win probability, run line, or moneyline.\n"
    "   - EXCLUDE all betting odds, market consensus, and line movements.\n\n"
    "**Fundamental Data Parameters**\n"
    "- **Starting Pitching:** Focus exclusively on predictive baselines such as xFIP, SIERA, "
    "K-BB%, and platoon/handedness splits. Ignore traditional ERA.\n"
    "- **Bullpen:** Assess high-leverage reliever fatigue over the preceding 72 hours and "
    "aggregate bullpen xFIP.\n"
    "- **Offense:** Evaluate team wRC+, OPS, and ISO, specifically isolated against the "
    "handedness of the opposing starting pitcher.\n"
    "- **Environment:** Factor in ballpark dimensions, stadium run modifiers, wind direction, "
    "and air density/temperature.\n\n"
    "**Output Format (STRICT):** Output ONLY a valid JSON object with EXACTLY these fields:\n"
    "{\n"
    '  "away_team": "<string>",\n'
    '  "home_team": "<string>",\n'
    '  "match_name": "<string>",\n'
    '  "away_pitcher": {"name": "<string>", "handedness": "L or R (never null)", "xfip": <float|null>, "siera": <float|null>, "k_bb_pct": <float|null>, "vs_lhb": <float|null>, "vs_rhb": <float|null>},\n'
    '  "home_pitcher": {"name": "<string>", "handedness": "L or R (never null)", "xfip": <float|null>, "siera": <float|null>, "k_bb_pct": <float|null>, "vs_lhb": <float|null>, "vs_rhb": <float|null>},\n'
    '  "away_offense": {"wrc_plus": <float|null>, "iso": <float|null>, "ops": <float|null>},\n'
    '  "home_offense": {"wrc_plus": <float|null>, "iso": <float|null>, "ops": <float|null>},\n'
    '  "away_bullpen": {"rested_key_relievers": <int 0-5|null>, "aggregate_xfip": <float|null>, "fatigue_flag": "<fresh|moderate|taxed|null>"},\n'
    '  "home_bullpen": {"rested_key_relievers": <int 0-5|null>, "aggregate_xfip": <float|null>, "fatigue_flag": "<fresh|moderate|taxed|null>"},\n'
    '  "venue": {"park_factor": <float|null>, "hr_factor": <float|null>, "wind_speed_mph": <float|null>, "wind_direction": "<string|null>", "temp_f": <float|null>, "stadium_name": "<string>"},\n'
    '  "confidence": <float 0-1>\n'
    "}\n\n"
    "**REMINDER:** Output ONLY the raw JSON. No markdown code blocks. No text before or after. "
    "Do NOT include a 'home_win', 'draw', or 'away_win' field — those are calculated downstream. "
    "Use null for any numeric field you cannot determine after exhaustive search (do not guess zero). "
    "IMPORTANT: handedness MUST be a string 'L' or 'R' — NEVER null. If unsure, default to 'R'."
)

USER_PROMPT_TEMPLATE = (
    "**Task:** Extract fundamental sabermetric inputs for today's MLB matchup.\n\n"
    "**Match Details:**\n"
    "- **Away Team:** {away_team}\n"
    "- **Home Team:** {home_team}\n"
    "- **Date:** {match_date}\n"
    "- **Away Pitcher:** {away_pitcher}\n"
    "- **Home Pitcher:** {home_pitcher}\n\n"
    "**Execution Instructions:**\n"
    "1. **Step 1 (Venue):** Resolve `{home_team}` to its official home ballpark name.\n"
    "2. **Step 2 (Sabermetrics Extraction):** Gather recent xFIP, K-BB%, SIERA, and "
    "handedness splits for both confirmed starters listed above.\n\n"
    "**Required Structural Output:**\n"
    "- **Venue:** [Official Stadium Name]\n"
    "- **Starting Pitchers:** [Away Pitcher Name] ([L/R]) vs. [Home Pitcher Name] ([L/R])\n"
    "- **Starter Metrics:** [xFIP / K-BB% / Platoon Splits for both pitchers]\n"
    "- **Bullpen Rest Index:** [High-leverage usage past 72h]\n"
    "- **Park Factor Modifiers:** [Run/HR park factors for resolved stadium]\n\n"
    "Output ONLY the raw JSON object — no markdown, no explanation."
)

# ── MLB Stats API ───────────────────────────────────────────

def fetch_probable_pitchers(date_str: str) -> dict:
    """
    Fetch probable pitchers from MLB Stats API for a given date (YYYY-MM-DD).
    Returns a dict keyed by (away_team_lower, home_team_lower) -> (away_pitcher, home_pitcher).
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    result = {}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            if resp.status != 200:
                print(f"[MLB API] HTTP {resp.status} for {date_str}", file=sys.stderr)
                return result
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except Exception as e:
        print(f"[MLB API] Failed to fetch probable pitchers for {date_str}: {e}", file=sys.stderr)
        return result

    total_games = sum(len(d.get("games", [])) for d in data.get("dates", []))
    print(f"[MLB API] {total_games} games found for {date_str}", file=sys.stderr)

    for d in data.get("dates", []):
        for game in d.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            away_team = (away.get("team", {}).get("name") or "").lower()
            home_team = (home.get("team", {}).get("name") or "").lower()
            away_pitcher = away.get("probablePitcher", {}).get("fullName", "TBD") if away.get("probablePitcher") else "TBD"
            home_pitcher = home.get("probablePitcher", {}).get("fullName", "TBD") if home.get("probablePitcher") else "TBD"

            if away_team and home_team:
                result[(away_team, home_team)] = (away_pitcher, home_pitcher)
                print(f"[MLB API]   {away_team} @ {home_team}: {away_pitcher} vs {home_pitcher}", file=sys.stderr)

    print(f"[MLB API] Loaded {len(result)} matchups for {date_str}", file=sys.stderr)
    return result


def lookup_pitchers(away_team: str, home_team: str, date_str: str) -> tuple[str, str]:
    """
    Look up probable pitchers for a matchup. Tries exact match first,
    then partial (substring) match as fallback.
    Returns (away_pitcher_name, home_pitcher_name).
    """
    ymd = _extract_date_ymd(date_str)
    if ymd == "Unknown":
        print(f"[MLB API] Cannot look up pitchers — unknown date", file=sys.stderr)
        return ("TBD", "TBD")

    pitchers = fetch_probable_pitchers(ymd)
    away_lower = away_team.lower().strip()
    home_lower = home_team.lower().strip()

    # Exact match
    key = (away_lower, home_lower)
    if key in pitchers:
        print(f"[MLB API] Exact match: '{away_lower}' @ '{home_lower}'", file=sys.stderr)
        return pitchers[key]

    # Partial match — check if team names are substrings
    for (a, h), (ap, hp) in pitchers.items():
        if (away_lower in a or a in away_lower) and (home_lower in h or h in home_lower):
            print(f"[MLB API] Partial match: '{away_lower}'->'{a}', '{home_lower}'->'{h}'", file=sys.stderr)
            return (ap, hp)

    # Try swapping (some APIs list home first)
    for (a, h), (ap, hp) in pitchers.items():
        if (away_lower in h or h in away_lower) and (home_lower in a or a in home_lower):
            print(f"[MLB API] Swapped match: '{away_lower}'->'{h}', '{home_lower}'->'{a}'", file=sys.stderr)
            return (hp, ap)

    print(f"[MLB API] No match for '{away_lower}' @ '{home_lower}' on {ymd}. Available: {list(pitchers.keys())}", file=sys.stderr)
    return ("TBD", "TBD")


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


def _parse_json_array(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def extract_mlb_markets_from_event(event: dict) -> list[dict]:
    """
    Extract moneyline markets from a Gamma event.
    Returns a list of dicts with type, token_id, question, team_name.
    Also returns the game date extracted from the moneyline market's gameStartTime.
    """
    markets = event.get("markets", [])
    title = (event.get("title") or "").lower()

    print(f"[MLB Markets] Title: '{title}'", file=sys.stderr)
    print(f"[MLB Markets] ===== ALL {len(markets)} MARKETS =====", file=sys.stderr)
    for i, m in enumerate(markets):
        q = m.get("question", "NO_QUESTION")
        mt = m.get("sportsMarketType", "none")
        gst = m.get("gameStartTime", "none")
        print(f"[MLB Markets]   [{i}] type={mt} gameStartTime={gst} q='{q[:80]}'", file=sys.stderr)

    result = []
    game_date = "Unknown"

    for i, m in enumerate(markets):
        market_type = m.get("sportsMarketType", "")
        if market_type != "moneyline":
            print(f"[MLB Markets]   [{i}] SKIP (type={market_type})", file=sys.stderr)
            continue

        clob_ids = _parse_json_array(m.get("clobTokenIds", "[]"))
        if len(clob_ids) < 2:
            print(f"[MLB Markets]   [{i}] SKIP (not enough clobTokenIds: {len(clob_ids)})", file=sys.stderr)
            continue

        outcomes = _parse_json_array(m.get("outcomes", "[]"))
        if len(outcomes) < 2:
            print(f"[MLB Markets]   [{i}] SKIP (not enough outcomes: {len(outcomes)})", file=sys.stderr)
            continue

        # Extract game date from market's gameStartTime (NOT event startDate — that's market creation date)
        gst = m.get("gameStartTime", "")
        if gst:
            game_date = _extract_date_ymd(gst)
            print(f"[MLB Markets]   [{i}] Game date from gameStartTime: {game_date}", file=sys.stderr)

        away_team = outcomes[0]
        home_team = outcomes[1]
        away_token = clob_ids[0]
        home_token = clob_ids[1]

        result.append({
            "type": "away",
            "token_id": away_token,
            "question": m.get("question", ""),
            "team_name": away_team,
        })
        result.append({
            "type": "home",
            "token_id": home_token,
            "question": m.get("question", ""),
            "team_name": home_team,
        })

        print(f"[MLB Markets]   [{i}] -> MONEYLINE: away='{away_team}' home='{home_team}' date={game_date}", file=sys.stderr)
        break

    print(f"[MLB Markets] Total: {len(result)} (away={'yes' if any(r['type']=='away' for r in result) else 'NO'}, home={'yes' if any(r['type']=='home' for r in result) else 'NO'}) game_date={game_date}", file=sys.stderr)
    return result, game_date


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
                return float(data.get("price", "0"))
    except Exception as e:
        print(f"[CLOB] Price fetch failed for token {token_id}: {e}", file=sys.stderr)
    return None


def share_price_to_decimal_odds(price: float) -> float:
    return 1.0 / price if price > 0 else float("inf")


# ── OpenAI ──────────────────────────────────────────────────

def get_openai_client() -> AsyncOpenAI:
    if not SURPLUS_API_KEY:
        raise RuntimeError("SURPLUS_API_KEY not set")
    return AsyncOpenAI(api_key=SURPLUS_API_KEY, base_url=SURPLUS_BASE_URL)


def _extract_json_from_text(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("LLM returned empty response")
    cleaned = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        preview = cleaned[:300] if len(cleaned) > 300 else cleaned
        raise ValueError(f"LLM response contains no JSON object. Raw: {preview}")
    cleaned = cleaned[start:end+1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        preview = cleaned[:300] if len(cleaned) > 300 else cleaned
        raise ValueError(f"Invalid JSON from LLM: {e}. Content: {preview}")


def _extract_content_from_choice(choice) -> Optional[str]:
    if not choice or not choice.message:
        return None
    msg = choice.message
    content = msg.content
    if content and content.strip():
        return content
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning and reasoning.strip():
        return reasoning
    for attr in ["reasoning", "thinking", "thought"]:
        val = getattr(msg, attr, None)
        if val and val.strip():
            return val
    return None


async def fetch_sabermetric_variables(
    match_title: str, away_team: str, home_team: str,
    match_date: str = "Unknown",
    away_pitcher: str = "TBD",
    home_pitcher: str = "TBD",
) -> SabermetricOutput:
    client = get_openai_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        away_team=away_team, home_team=home_team,
        match_date=match_date,
        away_pitcher=away_pitcher,
        home_pitcher=home_pitcher,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # Reasoning models (gpt-5.x) spend ~3000 tokens thinking internally.
    # 8000 max_tokens gives them 5000+ tokens of headroom for the JSON output.
    MAX_TOKENS = 8000
    raw = None
    last_error = None
    data = None

    for attempt, (mode_label, kwargs) in enumerate([
        ("raw", {"temperature": 0.1, "max_tokens": MAX_TOKENS}),
        ("json_object", {"temperature": 0.1, "max_tokens": MAX_TOKENS, "response_format": {"type": "json_object"}}),
    ]):
        try:
            response = await client.chat.completions.create(model=SURPLUS_MODEL, messages=messages, **kwargs)
            if response.choices:
                choice = response.choices[0]
                raw = _extract_content_from_choice(choice)
                if raw and raw.strip():
                    data = _extract_json_from_text(raw)
                    break
                finish = choice.finish_reason
                if finish == "length":
                    last_error = ValueError(f"{mode_label} mode: finish_reason=length (max_tokens={MAX_TOKENS} too low for reasoning model)")
                else:
                    last_error = ValueError(f"{mode_label} mode returned empty (finish_reason={finish})")
            else:
                last_error = ValueError(f"{mode_label} mode: no choices")
        except ValueError as e:
            last_error = e
            if raw and raw.strip():
                break
        except Exception as e:
            last_error = e

    if data is None:
        raise ValueError(f"All API modes failed. Model: {SURPLUS_MODEL}. Last error: {last_error}")

    return SabermetricOutput(
        away_team=str(data.get("away_team", away_team)),
        home_team=str(data.get("home_team", home_team)),
        match_name=str(data.get("match_name", match_title)),
        away_pitcher=PitcherData(**_strip_none(data.get("away_pitcher", {}))) if isinstance(data.get("away_pitcher"), dict) else PitcherData(),
        home_pitcher=PitcherData(**_strip_none(data.get("home_pitcher", {}))) if isinstance(data.get("home_pitcher"), dict) else PitcherData(),
        away_offense=OffenseData(**_strip_none(data.get("away_offense", {}))) if isinstance(data.get("away_offense"), dict) else OffenseData(),
        home_offense=OffenseData(**_strip_none(data.get("home_offense", {}))) if isinstance(data.get("home_offense"), dict) else OffenseData(),
        away_bullpen=BullpenData(**_strip_none(data.get("away_bullpen", {}))) if isinstance(data.get("away_bullpen"), dict) else BullpenData(),
        home_bullpen=BullpenData(**_strip_none(data.get("home_bullpen", {}))) if isinstance(data.get("home_bullpen"), dict) else BullpenData(),
        venue=VenueData(**_strip_none(data.get("venue", {}))) if isinstance(data.get("venue"), dict) else VenueData(),
        confidence=float(data.get("confidence", 0.5)),
    )


# ── Probability Calculation (Python only) ───────────────────

def sabermetrics_to_probabilities(sm: SabermetricOutput) -> dict:
    league_avg_xfip = 4.20
    league_avg_wrc_plus = 100.0
    league_avg_iso = 0.155
    league_avg_k_bb = 14.0

    away_xfip = sm.away_pitcher.xfip or league_avg_xfip
    home_xfip = sm.home_pitcher.xfip or league_avg_xfip
    away_kbb = sm.away_pitcher.k_bb_pct or league_avg_k_bb
    home_kbb = sm.home_pitcher.k_bb_pct or league_avg_k_bb
    away_wrc = sm.away_offense.wrc_plus or league_avg_wrc_plus
    home_wrc = sm.home_offense.wrc_plus or league_avg_wrc_plus
    away_iso = sm.away_offense.iso or league_avg_iso
    home_iso = sm.home_offense.iso or league_avg_iso

    away_pitching_edge = (league_avg_xfip - home_xfip) * 0.08
    home_pitching_edge = (league_avg_xfip - away_xfip) * 0.08
    away_kbb_edge = ((away_kbb - league_avg_k_bb) / league_avg_k_bb) * 0.06
    home_kbb_edge = ((home_kbb - league_avg_k_bb) / league_avg_k_bb) * 0.06
    away_offense_edge = ((away_wrc - league_avg_wrc_plus) / league_avg_wrc_plus) * 0.10
    home_offense_edge = ((home_wrc - league_avg_wrc_plus) / league_avg_wrc_plus) * 0.10
    away_iso_edge = ((away_iso - league_avg_iso) / league_avg_iso) * 0.04
    home_iso_edge = ((home_iso - league_avg_iso) / league_avg_iso) * 0.04

    away_bullpen_edge = 0.0
    home_bullpen_edge = 0.0
    if sm.away_bullpen.aggregate_xfip is not None:
        away_bullpen_edge = (league_avg_xfip - sm.away_bullpen.aggregate_xfip) * 0.04
    if sm.home_bullpen.aggregate_xfip is not None:
        home_bullpen_edge = (league_avg_xfip - sm.home_bullpen.aggregate_xfip) * 0.04

    fatigue_map = {"fresh": 0.005, "moderate": -0.01, "taxed": -0.025}
    if sm.away_bullpen.fatigue_flag:
        away_bullpen_edge += fatigue_map.get(sm.away_bullpen.fatigue_flag.lower(), 0.0)
    if sm.home_bullpen.fatigue_flag:
        home_bullpen_edge += fatigue_map.get(sm.home_bullpen.fatigue_flag.lower(), 0.0)

    home_field_advantage = 0.04
    park_factor = sm.venue.park_factor or 100.0
    park_modifier = ((park_factor - 100.0) / 100.0) * 0.03
    hr_factor = sm.venue.hr_factor or 100.0
    hr_modifier = ((hr_factor - 100.0) / 100.0) * 0.02
    away_hr_edge = hr_modifier * ((away_iso - league_avg_iso) / league_avg_iso)
    home_hr_edge = hr_modifier * ((home_iso - league_avg_iso) / league_avg_iso)

    away_edge = (away_pitching_edge + away_kbb_edge + away_offense_edge + away_iso_edge
               + away_bullpen_edge + away_hr_edge - home_field_advantage - park_modifier)
    home_edge = (home_pitching_edge + home_kbb_edge + home_offense_edge + home_iso_edge
               + home_bullpen_edge + home_hr_edge + home_field_advantage + park_modifier)

    def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
    net_edge = away_edge - home_edge
    away_prob = sigmoid(math.log(0.46 / 0.54) + net_edge * 3.0)
    home_prob = 1.0 - away_prob
    away_prob = max(0.25, min(0.75, away_prob))
    home_prob = 1.0 - away_prob

    parts = []
    if sm.away_pitcher.xfip and sm.home_pitcher.xfip:
        parts.append(f"Pitching: {sm.away_pitcher.name} xFIP {sm.away_pitcher.xfip:.2f} vs {sm.home_pitcher.name} xFIP {sm.home_pitcher.xfip:.2f}")
    if sm.away_offense.wrc_plus and sm.home_offense.wrc_plus:
        parts.append(f"Offense: {sm.away_team} wRC+ {sm.away_offense.wrc_plus:.0f} vs {sm.home_team} wRC+ {sm.home_offense.wrc_plus:.0f}")
    if sm.venue.park_factor:
        parts.append(f"Park: {sm.venue.stadium_name} factor {sm.venue.park_factor:.0f}")
    if sm.away_bullpen.fatigue_flag or sm.home_bullpen.fatigue_flag:
        parts.append(f"Bullpen: AWAY {sm.away_bullpen.fatigue_flag or 'normal'} | HOME {sm.home_bullpen.fatigue_flag or 'normal'}")
    justification = " | ".join(parts) if parts else "Insufficient sabermetric data"
    return {"away_prob": away_prob, "home_prob": home_prob, "justification": justification}


def calculate_ev(probs: dict, clob_prices: dict) -> dict:
    results = {
        "away": {"prob": probs["away_prob"], "price": None, "odds": None, "ev": None, "token_id": None},
        "home": {"prob": probs["home_prob"], "price": None, "odds": None, "ev": None, "token_id": None},
    }
    for outcome in ["away", "home"]:
        pd = clob_prices.get(outcome)
        if pd and pd.get("price") is not None:
            price = pd["price"]
            results[outcome]["price"] = price
            results[outcome]["token_id"] = pd.get("token_id")
            if price > 0:
                odds = share_price_to_decimal_odds(price)
                results[outcome]["odds"] = odds
                results[outcome]["ev"] = (results[outcome]["prob"] * odds) - 1.0
    return results


def calculate_kelly_bet(ev: float, prob: float, odds: float, bankroll: float, kelly_fraction: float, threshold: float) -> Optional[dict]:
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

def get_ev_threshold(gid): return ev_thresholds.get(gid, DEFAULT_EV_THRESHOLD)
def set_ev_threshold(gid, t): ev_thresholds[gid] = t
def get_kelly_fraction(gid): return kelly_fractions.get(gid, DEFAULT_KELLY_FRACTION)
def set_kelly_fraction(gid, f): kelly_fractions[gid] = f
def get_bankroll(gid): return bankrolls.get(gid, DEFAULT_BANKROLL)
def set_bankroll(gid, b): bankrolls[gid] = b


@bot.command(name="mlb")
async def cmd_mlb(ctx, *, args: str = ""):
    print(f"[MLB CMD] Received !mlb from {ctx.author} with args: '{args}'", file=sys.stderr)

    if not args:
        await ctx.send(
            "**MLB Sabermetric Bot — Kelly Criterion Mode**\n"
            "`!mlb <polymarket_url>` — Analyze MLB match, show EV + Kelly stakes\n"
            "`!mlb threshold <0.05>` — Set EV threshold\n"
            "`!mlb kelly <0.5>` — Set Kelly fraction\n"
            "`!mlb bankroll <1000>` — Set bankroll size\n"
            "`!mlbstatus` — Show current config\n\n"
            "**How it works:**\n"
            "1. Python scrapes probable pitchers from MLB Stats API\n"
            "2. AI extracts sabermetric variables (xFIP, wRC+, bullpen, park factors)\n"
            "3. Python calculates true win probability from those variables\n"
            "4. Kelly Criterion determines optimal stakes for +EV outcomes"
        )
        return

    try:
        if args.lower().startswith("threshold"):
            parts = args.split()
            if len(parts) >= 2:
                gid = ctx.guild.id if ctx.guild else ctx.author.id
                set_ev_threshold(gid, float(parts[1]))
                await ctx.send(f"✅ EV threshold set to **{float(parts[1])*100:.1f}%**")
            return

        if args.lower().startswith("kelly"):
            parts = args.split()
            if len(parts) >= 2:
                v = float(parts[1])
                if v <= 0 or v > 1.0:
                    await ctx.send("❌ Kelly fraction must be between 0.01 and 1.0"); return
                gid = ctx.guild.id if ctx.guild else ctx.author.id
                set_kelly_fraction(gid, v)
                label = "full-Kelly" if v == 1.0 else f"{v*100:.0f}%-Kelly"
                await ctx.send(f"✅ Kelly fraction set to **{v}** ({label})")
            return

        if args.lower().startswith("bankroll"):
            parts = args.split()
            if len(parts) >= 2:
                v = float(parts[1])
                if v <= 0:
                    await ctx.send("❌ Bankroll must be positive"); return
                gid = ctx.guild.id if ctx.guild else ctx.author.id
                set_bankroll(gid, v)
                await ctx.send(f"✅ Bankroll set to **${v:,.2f}**")
            return

        url = args.strip()
        slug = parse_polymarket_url(url)
        if not slug:
            await ctx.send("❌ Could not extract event slug from URL."); return

        async with ctx.typing():
            event = await asyncio.to_thread(fetch_event_from_gamma, slug)
            if not event:
                await ctx.send(f"❌ Event not found for slug `{slug}`."); return

            title = event.get("title", slug)

            markets, game_date = extract_mlb_markets_from_event(event)

            if len(markets) < 2:
                await ctx.send(
                    f"❌ Could not find enough MLB moneyline markets (away/home). Found: {len(markets)}.\n"
                    f"Event: `{title}`\nMarkets in event: {len(event.get('markets', []))}\n"
                    f"Check server logs for full market dump."
                )
                return

            market_lookup = {}
            for m in markets:
                if m["type"] not in market_lookup:
                    market_lookup[m["type"]] = m

            if "away" not in market_lookup or "home" not in market_lookup:
                await ctx.send(
                    f"❌ Found markets but not both away/home. Away: {'yes' if 'away' in market_lookup else 'NO'}, "
                    f"Home: {'yes' if 'home' in market_lookup else 'NO'}."
                )
                return

            away_team = market_lookup["away"].get("team_name", "Away")
            home_team = market_lookup["home"].get("team_name", "Home")

            series = event.get("series", [])
            league = series[0].get("title", "MLB") if isinstance(series, list) and series else "MLB"

            # ── Scrape probable pitchers from MLB Stats API ──
            print(f"[MLB CMD] Looking up pitchers: {away_team} @ {home_team} on {game_date}", file=sys.stderr)
            away_pitcher_name, home_pitcher_name = await asyncio.to_thread(
                lookup_pitchers, away_team, home_team, game_date
            )
            print(f"[MLB CMD] Pitchers: {away_team} -> {away_pitcher_name}, {home_team} -> {home_pitcher_name}", file=sys.stderr)

            clob_prices = {}
            for outcome_type in ["away", "home"]:
                m = market_lookup.get(outcome_type)
                if m:
                    price = await asyncio.to_thread(fetch_clob_best_ask, m["token_id"])
                    clob_prices[outcome_type] = {"price": price, "token_id": m["token_id"], "question": m.get("question", "")}

            try:
                sm = await fetch_sabermetric_variables(
                    match_title=title, away_team=away_team, home_team=home_team,
                    match_date=game_date,
                    away_pitcher=away_pitcher_name,
                    home_pitcher=home_pitcher_name,
                )
            except Exception as e:
                await ctx.send(f"❌ **Sabermetric Engine Error**: {type(e).__name__}: {e}")
                return

            probs = sabermetrics_to_probabilities(sm)
            ev_results = calculate_ev(probs, clob_prices)
            gid = ctx.guild.id if ctx.guild else ctx.author.id
            threshold = get_ev_threshold(gid)
            kelly_frac = get_kelly_fraction(gid)
            bankroll = get_bankroll(gid)
            kelly_label = "Full-Kelly" if kelly_frac == 1.0 else f"{kelly_frac*100:.0f}%-Kelly"

            embed = discord.Embed(
                title=f"⚾ MLB Sabermetric Analysis: {sm.match_name}",
                description=(f"**League:** {league} | **Date:** {game_date}\n"
                           f"**Venue:** {sm.venue.stadium_name}\n"
                           f"**Staking:** {kelly_label} | **Bankroll:** ${bankroll:,.2f} | **Threshold:** {threshold*100:.1f}%"),
                color=discord.Color.dark_green(),
            )

            pitcher_lines = []
            p1_name = f"{sm.away_pitcher.name} ({sm.away_pitcher.handedness})" if sm.away_pitcher.handedness else sm.away_pitcher.name
            p2_name = f"{sm.home_pitcher.name} ({sm.home_pitcher.handedness})" if sm.home_pitcher.handedness else sm.home_pitcher.name
            if sm.away_pitcher.xfip and sm.home_pitcher.xfip:
                p1 = f"**{p1_name}** (AWAY): xFIP {sm.away_pitcher.xfip:.2f}"
                if sm.away_pitcher.siera: p1 += f" | SIERA {sm.away_pitcher.siera:.2f}"
                if sm.away_pitcher.k_bb_pct: p1 += f" | K-BB% {sm.away_pitcher.k_bb_pct:.1f}%"
                pitcher_lines.append(p1)
                p2 = f"**{p2_name}** (HOME): xFIP {sm.home_pitcher.xfip:.2f}"
                if sm.home_pitcher.siera: p2 += f" | SIERA {sm.home_pitcher.siera:.2f}"
                if sm.home_pitcher.k_bb_pct: p2 += f" | K-BB% {sm.home_pitcher.k_bb_pct:.1f}%"
                pitcher_lines.append(p2)
            else:
                pitcher_lines.append(f"{p1_name} vs {p2_name}")
            embed.add_field(name="🎯 Starting Pitchers", value="\n".join(pitcher_lines), inline=False)

            offense_lines = []
            if sm.away_offense.wrc_plus and sm.home_offense.wrc_plus:
                o1 = f"**{sm.away_team}**: wRC+ {sm.away_offense.wrc_plus:.0f}"
                if sm.away_offense.iso: o1 += f" | ISO {sm.away_offense.iso:.3f}"
                if sm.away_offense.ops: o1 += f" | OPS {sm.away_offense.ops:.3f}"
                offense_lines.append(o1)
                o2 = f"**{sm.home_team}**: wRC+ {sm.home_offense.wrc_plus:.0f}"
                if sm.home_offense.iso: o2 += f" | ISO {sm.home_offense.iso:.3f}"
                if sm.home_offense.ops: o2 += f" | OPS {sm.home_offense.ops:.3f}"
                offense_lines.append(o2)
            if offense_lines:
                embed.add_field(name="⚡ Offensive Efficiency", value="\n".join(offense_lines), inline=False)

            bullpen_lines = []
            for _, team_name, bp in [("away", sm.away_team, sm.away_bullpen), ("home", sm.home_team, sm.home_bullpen)]:
                if bp.fatigue_flag:
                    line = f"**{team_name}**: {bp.rested_key_relievers or '?'} rested"
                    if bp.aggregate_xfip: line += f" | Bullpen xFIP {bp.aggregate_xfip:.2f}"
                    line += f" | **{bp.fatigue_flag.upper()}**"
                    bullpen_lines.append(line)
            if bullpen_lines:
                embed.add_field(name="🫀 Bullpen Availability (72hr)", value="\n".join(bullpen_lines), inline=False)

            venue_lines = []
            if sm.venue.park_factor:
                v = f"Park Factor: **{sm.venue.park_factor:.0f}**"
                if sm.venue.hr_factor: v += f" (HR: {sm.venue.hr_factor:.0f})"
                venue_lines.append(v)
            if sm.venue.wind_speed_mph:
                venue_lines.append(f"Wind: {sm.venue.wind_speed_mph} mph {sm.venue.wind_direction or ''}")
            if sm.venue.temp_f:
                venue_lines.append(f"Temp: {sm.venue.temp_f:.0f}°F")
            if venue_lines:
                embed.add_field(name=f"🏟️ Venue — {sm.venue.stadium_name}", value="\n".join(venue_lines), inline=False)

            embed.add_field(name="📊 Python Win Probabilities",
                          value=f"🚶 **{sm.away_team}**: {probs['away_prob']*100:.1f}%\n🏠 **{sm.home_team}**: {probs['home_prob']*100:.1f}%",
                          inline=False)
            embed.add_field(name="📐 Justification", value=probs["justification"], inline=False)

            ev_lines, kelly_lines, positive_ev = [], [], []
            for outcome, label, emoji in [("away", sm.away_team, "🚶"), ("home", sm.home_team, "🏠")]:
                r = ev_results[outcome]
                ps = f"{r['price']*100:.1f}¢" if r["price"] is not None else "N/A"
                os_ = f"{r['odds']:.2f}" if r["odds"] is not None else "N/A"
                es = f"{r['ev']*100:+.1f}%" if r["ev"] is not None else "N/A"
                if r["ev"] is not None and r["ev"] >= threshold and r["odds"] is not None:
                    ev_lines.append(f"✅ {emoji} **{label}**: Price {ps} | Odds {os_} | EV **{es}**")
                    positive_ev.append(outcome)
                    k = calculate_kelly_bet(r["ev"], r["prob"], r["odds"], bankroll, kelly_frac, threshold)
                    if k:
                        kelly_lines.append(f"{emoji} **{label}**: Stake **${k['stake']:,.2f}** ({k['scaled_kelly_pct']:.1f}%)\n"
                                         f"　↳ Full Kelly: {k['full_kelly_pct']:.1f}% | Exp profit: **${k['expected_profit']:,.2f}** ({k['expected_roi_pct']:+.1f}% ROI)")
                elif r["ev"] is not None:
                    ev_lines.append(f"❌ {emoji} **{label}**: Price {ps} | Odds {os_} | EV {es}")
                else:
                    ev_lines.append(f"⚪ {emoji} **{label}**: Price {ps} | No EV data")

            embed.add_field(name="CLOB Prices & EV", value="\n".join(ev_lines) or "No EV data", inline=False)
            if kelly_lines:
                embed.add_field(name=f"💰 Kelly Stakes", value="\n".join(kelly_lines), inline=False)
            embed.add_field(name="AI Confidence", value=f"{sm.confidence*100:.0f}%", inline=True)

            if positive_ev:
                total_stake = sum(calculate_kelly_bet(ev_results[o]["ev"], ev_results[o]["prob"], ev_results[o]["odds"], bankroll, kelly_frac, threshold)["stake"] for o in positive_ev)
                total_exp = sum(calculate_kelly_bet(ev_results[o]["ev"], ev_results[o]["prob"], ev_results[o]["odds"], bankroll, kelly_frac, threshold)["expected_profit"] for o in positive_ev)
                embed.add_field(name="📋 Bet Summary", value=f"**{len(positive_ev)}** +EV outcome(s)\nTotal stake: **${total_stake:,.2f}** ({(total_stake/bankroll)*100:.1f}%)\nTotal exp profit: **${total_exp:,.2f}**", inline=True)
            else:
                embed.add_field(name="📋 Bet Summary", value="No +EV outcomes above threshold.", inline=True)

            embed.set_footer(text=f"Engine: {SURPLUS_MODEL} | Kelly: f*=EV/(odds−1)×{kelly_frac} | Max 25%/bet")
            await ctx.send(embed=embed)

    except Exception as e:
        print(f"[MLB CMD] UNHANDLED ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            await ctx.send(f"❌ **Unexpected Error**: {type(e).__name__}: {e}")
        except Exception:
            pass


@bot.command(name="mlbstatus")
async def cmd_mlb_status(ctx):
    gid = ctx.guild.id if ctx.guild else ctx.author.id
    api_status = "✅" if SURPLUS_API_KEY else "❌"
    kelly_frac = get_kelly_fraction(gid)
    kelly_label = "Full-Kelly" if kelly_frac == 1.0 else f"{kelly_frac*100:.0f}%-Kelly"
    await ctx.send(embed=discord.Embed(
        title="⚾ MLB Bot Status",
        description="\n".join([
            f"**Mode:** MLB Sabermetric + Kelly Criterion",
            f"**EV Threshold:** {get_ev_threshold(gid)*100:.1f}%",
            f"**Kelly Fraction:** {kelly_frac} ({kelly_label})",
            f"**Bankroll:** ${get_bankroll(gid):,.2f}",
            f"**AI Model:** {SURPLUS_MODEL}",
            f"**Surplus API:** {api_status}",
        ]),
        color=discord.Color.green() if SURPLUS_API_KEY else discord.Color.orange(),
    ))


@bot.event
async def on_ready():
    print(f"[MLB Bot] Online: {bot.user.name} ({bot.user.id}) | Model: {SURPLUS_MODEL}", file=sys.stderr)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    print(f"[MLB Bot Error] {type(error).__name__}: {error}", file=sys.stderr)
    try:
        await ctx.send(f"❌ **Command Error**: {type(error).__name__}: {error}")
    except Exception:
        pass


async def main():
    if not DISCORD_TOKEN:
        print("FATAL: MLB_BOT_TOKEN or DISCORD_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    async with bot:
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def shutdown():
        for task in asyncio.all_tasks(loop):
            task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try: loop.add_signal_handler(sig, shutdown)
        except NotImplementedError: pass
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print("[MLB Bot] Stopped cleanly", file=sys.stderr)