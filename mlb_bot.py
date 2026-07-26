"""
mlb_bot.py — MLB Sabermetric EV Betting Bot (Kelly Criterion Mode)
=====================================================================
Architecture:
  1. Gamma API         → fetch MLB market data + clobTokenIds
  2. OpenAI (gpt-5.4)  → isolated sabermetric variables (NOT probabilities)
  3. Python Backend    → deterministic probability calculation from sabermetrics
  4. CLOB API          → live best-ask prices → decimal odds
  5. EV Math           → EV = (prob * odds) - 1
  6. Kelly Staking     → f* = EV / (odds - 1), scaled by kelly fraction

Strict separation: LLM outputs raw structural data only.
ALL final probability calculations, risk management, and execution
decisions are handled exclusively by Python machine code.
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
import math
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

# ── Pydantic Output Schema (Sabermetric — no probabilities) ─

class PitcherData(BaseModel):
    name: str = Field(default="Unknown")
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
    "You are the sabermetric analysis module of an automated sports prediction framework. "
    "Your strict, singular function is to evaluate MLB matchups and output fundamental, "
    "mathematically isolated baseball variables for downstream processing.\n\n"
    "**Execution Constraints**\n"
    "1. **No Final Probabilities:** You must never calculate, synthesize, or output a "
    "final true win probability, run line prediction, or moneyline suggestion.\n"
    "2. **Backend Handoff:** Your output serves strictly as the raw structural data feed. "
    "All final deterministic probability calculations, risk-management parameters, and "
    "execution decisions are handled exclusively by the backend machine code.\n"
    "3. **No Market Consensus:** Completely ignore and exclude all market consensus data, "
    "betting odds, public betting percentages, and trading volumes.\n\n"
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
    '  "away_pitcher": {\n'
    '    "name": "<string>", "xfip": <float>, "siera": <float>,\n'
    '    "k_bb_pct": <float>, "vs_lhb": <float>, "vs_rhb": <float>\n'
    '  },\n'
    '  "home_pitcher": {\n'
    '    "name": "<string>", "xfip": <float>, "siera": <float>,\n'
    '    "k_bb_pct": <float>, "vs_lhb": <float>, "vs_rhb": <float>\n'
    '  },\n'
    '  "away_offense": { "wrc_plus": <float>, "iso": <float>, "ops": <float> },\n'
    '  "home_offense": { "wrc_plus": <float>, "iso": <float>, "ops": <float> },\n'
    '  "away_bullpen": {\n'
    '    "rested_key_relievers": <int 0-5>,\n'
    '    "aggregate_xfip": <float>,\n'
    '    "fatigue_flag": "<fresh|moderate|taxed>"\n'
    '  },\n'
    '  "home_bullpen": {\n'
    '    "rested_key_relievers": <int 0-5>,\n'
    '    "aggregate_xfip": <float>,\n'
    '    "fatigue_flag": "<fresh|moderate|taxed>"\n'
    '  },\n'
    '  "venue": {\n'
    '    "park_factor": <float>,\n'
    '    "hr_factor": <float>,\n'
    '    "wind_speed_mph": <float>,\n'
    '    "wind_direction": "<string>",\n'
    '    "temp_f": <float>,\n'
    '    "stadium_name": "<string>"\n'
    '  },\n'
    '  "confidence": <float 0-1>\n'
    "}\n\n"
    "**REMINDER:** Output ONLY the raw JSON. No markdown code blocks. No text before or after. "
    "Do NOT include a 'home_win', 'draw', or 'away_win' field — those are calculated downstream. "
    "Use null for any numeric field you cannot determine (do not guess zero)."
)

USER_PROMPT_TEMPLATE = (
    "**Task:** Extract and isolate the fundamental sabermetric inputs for the following "
    "MLB matchup. Format the data structurally so it can be ingested by the backend "
    "calculation engine.\n\n"
    "**Matchup Details:**\n"
    "- **Away Team:** {away_team}\n"
    "- **Home Team:** {home_team}\n"
    "- **Date:** {match_date}\n"
    "- **Venue:** {stadium_name}\n"
    "- **Starting Pitchers:** {away_pitcher} vs. {home_pitcher}\n\n"
    "**Required Outputs:**\n"
    "1. **Starting Pitcher Baselines:** Provide recent xFIP, SIERA, K-BB%, and performance "
    "splits against left-handed/right-handed batters for both starters.\n"
    "2. **Offensive Efficiency:** List both teams' wRC+ and ISO over the last 14 days "
    "specifically against the respective starter's throwing arm (LHP vs RHP).\n"
    "3. **Bullpen Availability:** Detail the usage rates and rest days for the top three "
    "high-leverage relievers on both rosters over the last 72 hours.\n"
    "4. **Venue/Environmental Modifiers:** Identify the specific stadium run factors and "
    "current weather conditions (wind speed/direction, temperature) affecting ball flight today.\n\n"
    "**System Reminder:** Output only the isolated fundamental variables requested. "
    "Do not attempt to calculate a winner or generate a definitive probability output. "
    "Output ONLY the raw JSON object — no markdown, no explanation."
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


def _team_in_question(team_name: str, question: str) -> bool:
    """Check if team name appears as a word or meaningful substring in question."""
    if not team_name or not question:
        return False
    q = question.lower()
    name = team_name.lower().strip()
    # Direct substring match
    if name in q:
        return True
    # Try individual words (e.g. "Yankees" matches "New York Yankees")
    for word in name.split():
        if len(word) > 3 and word in q:
            return True
    # Try short/abbreviation forms (e.g. "NYY" for Yankees)
    return False


def extract_mlb_markets_from_event(event: dict) -> list[dict]:
    """
    Extract moneyline markets from an MLB Polymarket event.
    Handles multiple Gamma API question formats with robust matching.

    Common Polymarket MLB question formats:
      - "Will [Team A] win?" (moneyline)
      - "[Team A] to win?"
      - "Moneyline: [Team A]"
      - "[Team A] @ [Team B] — [Team A] win?"
    """
    markets = event.get("markets", [])
    title = (event.get("title") or "").lower()
    teams = event.get("teams", [])

    # Build team name lists
    team_names = []
    for t in teams:
        name = (t.get("name") or "").lower().strip()
        short = (t.get("shortName") or t.get("abbreviation") or "").lower().strip()
        if name:
            team_names.append(name)
        if short and short != name:
            team_names.append(short)

    away_name = team_names[0] if len(team_names) >= 1 else ""
    home_name = team_names[1] if len(team_names) >= 2 else ""

    # Parse title for team names as fallback
    title_away = title_home = ""
    if " vs " in title:
        parts = [p.strip() for p in title.split(" vs ")]
        if len(parts) >= 2:
            title_away, title_home = parts[0].lower(), parts[1].lower()
    elif " @ " in title:
        parts = [p.strip() for p in title.split(" @ ")]
        if len(parts) >= 2:
            title_away, title_home = parts[0].lower(), parts[1].lower()

    # ── DEBUG: dump all raw market data ──
    print(f"[MLB Markets] Event title: '{title}'", file=sys.stderr)
    print(f"[MLB Markets] Teams from Gamma: away='{away_name}', home='{home_name}'", file=sys.stderr)
    print(f"[MLB Markets] Title teams: away='{title_away}', home='{title_home}'", file=sys.stderr)
    print(f"[MLB Markets] Total markets in event: {len(markets)}", file=sys.stderr)
    for i, m in enumerate(markets):
        q = m.get("question", "NO_QUESTION")
        cids = m.get("clobTokenIds", "[]")
        print(f"[MLB Markets]   [{i}] question='{q}' clobTokenIds={cids}", file=sys.stderr)

    result = []
    for m in markets:
        question = (m.get("question") or "").lower().strip()
        if not question:
            continue

        clob_ids = m.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (json.JSONDecodeError, TypeError):
                clob_ids = []
        if not clob_ids or len(clob_ids) == 0:
            continue

        token_id = clob_ids[0]

        has_win = "win" in question
        if not has_win:
            # Also check "defeat" / "beat" patterns for moneyline
            has_win = any(w in question for w in ["defeat", "beat", "moneyline"])

        away_matches_gamma = _team_in_question(away_name, question) if away_name else False
        home_matches_gamma = _team_in_question(home_name, question) if home_name else False
        away_matches_title = _team_in_question(title_away, question) if title_away else False
        home_matches_title = _team_in_question(title_home, question) if title_home else False

        effective_away = away_matches_gamma or away_matches_title
        effective_home = home_matches_gamma or home_matches_title

        market_type = None

        # Strategy 1: clear win keyword + single team match
        if has_win and effective_away and not effective_home:
            market_type = "away"
        elif has_win and effective_home and not effective_away:
            market_type = "home"
        elif has_win and effective_away and effective_home:
            # Both teams mentioned — figure out which is the subject
            # Check "beat" / "defeat": "Home beat Away" → home is subject
            if any(w in question for w in ["beat", "defeat"]):
                # The subject (winner) comes before "beat"/"defeat"
                for w in ["beat", "defeat"]:
                    if w in question:
                        subject_part = question.split(w)[0].strip()
                        if _team_in_question(away_name, subject_part):
                            market_type = "away"
                        elif _team_in_question(home_name, subject_part):
                            market_type = "home"
                        elif _team_in_question(title_away, subject_part):
                            market_type = "away"
                        elif _team_in_question(title_home, subject_part):
                            market_type = "home"
                        break
            else:
                # "Will Away win vs Home?" → first team mentioned is the subject
                away_pos = question.find(away_name) if away_name and away_name in question else 999
                home_pos = question.find(home_name) if home_name and home_name in question else 999
                market_type = "away" if away_pos < home_pos else "home"

        # Strategy 2: fallback — try title team names
        if market_type is None and has_win:
            if away_matches_title and not home_matches_title:
                market_type = "away"
            elif home_matches_title and not away_matches_title:
                market_type = "home"

        # Strategy 3: brute force — match ANY team name from the full list
        if market_type is None and has_win:
            matched_teams = []
            for i, name in enumerate(team_names):
                if _team_in_question(name, question):
                    # Determine if this is away (even index) or home (odd index)
                    half = len(team_names) / 2
                    side = "away" if i < half else "home"
                    matched_teams.append(side)
            if len(set(matched_teams)) == 1:
                market_type = matched_teams[0]

        if market_type:
            result.append({
                "type": market_type,
                "token_id": token_id,
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
            })
            print(f"[MLB Markets]   → MATCHED: type={market_type} question='{m.get('question','')[:80]}'", file=sys.stderr)

    print(f"[MLB Markets] Total matched: {len(result)} (away={sum(1 for r in result if r['type']=='away')}, home={sum(1 for r in result if r['type']=='home')})", file=sys.stderr)
    return result


def extract_pitchers_from_event(event: dict) -> tuple:
    away_pitcher = "TBD"
    home_pitcher = "TBD"

    # Check event-level metadata
    for key in ["awayPitcher", "away_pitcher", "pitcherAway"]:
        val = event.get(key) or (event.get("metadata") or {}).get(key)
        if val and isinstance(val, str) and val.strip():
            away_pitcher = val.strip()
            break

    for key in ["homePitcher", "home_pitcher", "pitcherHome"]:
        val = event.get(key) or (event.get("metadata") or {}).get(key)
        if val and isinstance(val, str) and val.strip():
            home_pitcher = val.strip()
            break

    return away_pitcher, home_pitcher


def extract_stadium_from_event(event: dict) -> str:
    for key in ["venue", "stadium", "location"]:
        val = event.get(key)
        if isinstance(val, dict):
            for name_key in ["name", "stadium", "venue"]:
                n = val.get(name_key)
                if n and isinstance(n, str) and n.strip():
                    return n.strip()
        if isinstance(val, str) and val.strip():
            return val.strip()

    meta = event.get("metadata") or event.get("meta") or {}
    if isinstance(meta, dict):
        for key in ["venue", "stadium", "location", "ballpark"]:
            val = meta.get(key)
            if isinstance(val, dict):
                for name_key in ["name", "stadium"]:
                    n = val.get(name_key)
                    if n and isinstance(n, str) and n.strip():
                        return n.strip()
            if isinstance(val, str) and val.strip():
                return val.strip()

    return "Unknown"


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


# ── OpenAI Sabermetric Engine ───────────────────────────────

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
        print(f"[OpenAI] Using reasoning_content ({len(reasoning)} chars)", file=sys.stderr)
        return reasoning
    for attr in ["reasoning", "thinking", "thought"]:
        val = getattr(msg, attr, None)
        if val and val.strip():
            print(f"[OpenAI] Using {attr} field ({len(val)} chars)", file=sys.stderr)
            return val
    return None


async def fetch_sabermetric_variables(
    match_title: str, away_team: str, home_team: str,
    match_date: str = "Unknown", stadium_name: str = "Unknown",
    away_pitcher: str = "TBD", home_pitcher: str = "TBD",
) -> SabermetricOutput:
    client = get_openai_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        away_team=away_team, home_team=home_team,
        match_date=match_date, stadium_name=stadium_name,
        away_pitcher=away_pitcher, home_pitcher=home_pitcher,
    )
    print(f"[OpenAI] Prompt ({len(user_prompt)} chars)", file=sys.stderr)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    MAX_TOKENS = 4000
    raw = None
    last_error = None
    data = None

    for attempt, (mode_label, kwargs) in enumerate([
        ("raw", {"temperature": 0.1, "max_tokens": MAX_TOKENS}),
        ("json_object", {"temperature": 0.1, "max_tokens": MAX_TOKENS, "response_format": {"type": "json_object"}}),
    ]):
        try:
            print(f"[OpenAI] Attempt {attempt+1}: {mode_label} mode (max_tokens={MAX_TOKENS})", file=sys.stderr)
            response = await client.chat.completions.create(model=SURPLUS_MODEL, messages=messages, **kwargs)
            print(f"[OpenAI] Full response object:", file=sys.stderr)
            print(f"  model: {response.model}", file=sys.stderr)
            print(f"  usage: {response.usage}", file=sys.stderr)
            print(f"  choices count: {len(response.choices)}", file=sys.stderr)
            if response.choices:
                choice = response.choices[0]
                print(f"  finish_reason: {choice.finish_reason}", file=sys.stderr)
                if choice.message:
                    print(f"  message.content length: {len(choice.message.content) if choice.message.content else 0}", file=sys.stderr)
                    reasoning = getattr(choice.message, "reasoning_content", None)
                    print(f"  message.reasoning_content length: {len(reasoning) if reasoning else 0}", file=sys.stderr)
            raw = _extract_content_from_choice(choice) if response.choices else None
            if raw and raw.strip():
                data = _extract_json_from_text(raw)
                break
            else:
                if response.choices:
                    finish = choice.finish_reason
                    if finish == "length":
                        print(f"[OpenAI] {mode_label}: finish_reason=length — response truncated.", file=sys.stderr)
                        last_error = ValueError(f"{mode_label} mode: finish_reason=length")
                    else:
                        print(f"[OpenAI] {mode_label}: returned empty content (finish_reason={finish})", file=sys.stderr)
                        last_error = ValueError(f"{mode_label} mode returned empty response (finish_reason={finish})")
                else:
                    last_error = ValueError(f"{mode_label} mode: response has no choices")
        except ValueError as e:
            last_error = e
            if raw and raw.strip():
                break
        except Exception as e:
            last_error = e
            print(f"[OpenAI] {mode_label} API call failed: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if data is None:
        raise ValueError(f"All API modes failed. Model: {SURPLUS_MODEL}. Last error: {last_error}")

    return SabermetricOutput(
        away_team=str(data.get("away_team", away_team)),
        home_team=str(data.get("home_team", home_team)),
        match_name=str(data.get("match_name", match_title)),
        away_pitcher=PitcherData(**data.get("away_pitcher", {})) if isinstance(data.get("away_pitcher"), dict) else PitcherData(name=away_pitcher),
        home_pitcher=PitcherData(**data.get("home_pitcher", {})) if isinstance(data.get("home_pitcher"), dict) else PitcherData(name=home_pitcher),
        away_offense=OffenseData(**data.get("away_offense", {})) if isinstance(data.get("away_offense"), dict) else OffenseData(),
        home_offense=OffenseData(**data.get("home_offense", {})) if isinstance(data.get("home_offense"), dict) else OffenseData(),
        away_bullpen=BullpenData(**data.get("away_bullpen", {})) if isinstance(data.get("away_bullpen"), dict) else BullpenData(),
        home_bullpen=BullpenData(**data.get("home_bullpen", {})) if isinstance(data.get("home_bullpen"), dict) else BullpenData(),
        venue=VenueData(**data.get("venue", {})) if isinstance(data.get("venue"), dict) else VenueData(stadium_name=stadium_name),
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

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

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


@bot.command(name="mlb")
async def cmd_mlb(ctx, *, args: str = ""):
    print(f"[MLB CMD] Received !mlb from {ctx.author} with args: '{args}'", file=sys.stderr)

    if not args:
        help_text = (
            "**MLB Sabermetric Bot — Kelly Criterion Mode**\n"
            "`!mlb <polymarket_url>` — Analyze MLB match, show EV + Kelly stakes\n"
            "`!mlb threshold <0.05>` — Set EV threshold (e.g. 0.05 = 5%)\n"
            "`!mlb kelly <0.5>` — Set Kelly fraction (0.5 = half-Kelly)\n"
            "`!mlb bankroll <1000>` — Set bankroll size in USD\n"
            "`!mlbstatus` — Show current config\n\n"
            "**How it works:**\n"
            "1. AI extracts sabermetric variables (xFIP, wRC+, bullpen, park factors)\n"
            "2. Python calculates true win probability from those variables\n"
            "3. Kelly Criterion determines optimal stakes for +EV outcomes"
        )
        await ctx.send(help_text)
        return

    try:
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
                await ctx.send("❌ Invalid threshold. Use: `!mlb threshold 0.05`")
                return

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
                await ctx.send("❌ Invalid Kelly fraction. Use: `!mlb kelly 0.5`")
                return

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
                await ctx.send("❌ Invalid bankroll. Use: `!mlb bankroll 1000`")
                return

        url = args.strip()
        print(f"[MLB CMD] URL to analyze: '{url}'", file=sys.stderr)

        slug = parse_polymarket_url(url)
        if not slug:
            await ctx.send("❌ Could not extract event slug from URL.")
            return

        print(f"[MLB CMD] Extracted slug: '{slug}'", file=sys.stderr)

        async with ctx.typing():
            print(f"[MLB CMD] Fetching Gamma event for slug: {slug}", file=sys.stderr)
            event = await asyncio.to_thread(fetch_event_from_gamma, slug)
            if not event:
                await ctx.send(f"❌ Event not found for slug `{slug}`.")
                return

            print(f"[MLB CMD] Gamma event found: {event.get('title', '?')}", file=sys.stderr)

            title = event.get("title", slug)
            teams = event.get("teams", [])
            away_team = teams[0].get("name", "Away") if len(teams) >= 1 else "Away"
            home_team = teams[1].get("name", "Home") if len(teams) >= 2 else "Home"

            series = event.get("series", [])
            league = "MLB"
            if isinstance(series, list) and len(series) > 0:
                league = series[0].get("title", "MLB")

            match_date = event.get("startDate") or event.get("scheduledStart") or "Unknown"
            away_pitcher, home_pitcher = extract_pitchers_from_event(event)
            stadium_name = extract_stadium_from_event(event)

            # ── Extract markets with full debug logging ──
            markets = extract_mlb_markets_from_event(event)

            if len(markets) < 2:
                await ctx.send(
                    f"❌ Could not find enough MLB moneyline markets (away/home). Found: {len(markets)}.\n"
                    f"Event: `{title}`\n"
                    f"Markets in event: {len(event.get('markets', []))}\n"
                    f"Check the server logs for full market dump — the Gamma API question format may differ from what we expect.\n"
                    f"Paste the `[MLB Markets]` lines from `journalctl -u mlbbot` so I can fix the parser."
                )
                return

            market_lookup = {}
            for m in markets:
                if m["type"] not in market_lookup:
                    market_lookup[m["type"]] = m

            if "away" not in market_lookup or "home" not in market_lookup:
                await ctx.send(
                    f"❌ Found markets but not both away/home. Away: {'yes' if 'away' in market_lookup else 'NO'}, "
                    f"Home: {'yes' if 'home' in market_lookup else 'NO'}.\nCheck the logs for market dump."
                )
                return

            clob_prices = {}
            for outcome_type in ["away", "home"]:
                m = market_lookup.get(outcome_type)
                if m:
                    price = await asyncio.to_thread(fetch_clob_best_ask, m["token_id"])
                    clob_prices[outcome_type] = {
                        "price": price, "token_id": m["token_id"],
                        "question": m.get("question", ""),
                    }

            print(f"[MLB CMD] CLOB prices: away={clob_prices.get('away',{}).get('price')}, "
                  f"home={clob_prices.get('home',{}).get('price')}", file=sys.stderr)

            print(f"[MLB CMD] Calling sabermetric AI for: {title}", file=sys.stderr)
            try:
                sm = await fetch_sabermetric_variables(
                    match_title=title, away_team=away_team, home_team=home_team,
                    match_date=str(match_date), stadium_name=stadium_name,
                    away_pitcher=away_pitcher, home_pitcher=home_pitcher,
                )
            except Exception as e:
                await ctx.send(f"❌ **Sabermetric Engine Error**: {type(e).__name__}: {e}\nCheck SURPLUS_API_KEY and model `{SURPLUS_MODEL}`.")
                return

            print(f"[MLB CMD] Sabermetrics received: away_xFIP={sm.away_pitcher.xfip}, "
                  f"home_xFIP={sm.home_pitcher.xfip}, park_factor={sm.venue.park_factor}", file=sys.stderr)

            probs = sabermetrics_to_probabilities(sm)
            ev_results = calculate_ev(probs, clob_prices)
            gid = ctx.guild.id if ctx.guild else ctx.author.id
            threshold = get_ev_threshold(gid)
            kelly_frac = get_kelly_fraction(gid)
            bankroll = get_bankroll(gid)

            kelly_label = "Full-Kelly" if kelly_frac == 1.0 else f"{kelly_frac*100:.0f}%-Kelly"
            embed = discord.Embed(
                title=f"⚾ MLB Sabermetric Analysis: {sm.match_name}",
                description=(
                    f"**League:** {league} | **Date:** {match_date}\n"
                    f"**Venue:** {sm.venue.stadium_name}\n"
                    f"**Slug:** `{slug}`\n"
                    f"**Staking:** {kelly_label} | **Bankroll:** ${bankroll:,.2f} | **Threshold:** {threshold*100:.1f}%"
                ),
                color=discord.Color.dark_green(),
            )

            pitcher_lines = []
            if sm.away_pitcher.xfip and sm.home_pitcher.xfip:
                p1 = f"**{sm.away_pitcher.name}** (AWAY): xFIP {sm.away_pitcher.xfip:.2f}"
                if sm.away_pitcher.siera: p1 += f" | SIERA {sm.away_pitcher.siera:.2f}"
                if sm.away_pitcher.k_bb_pct: p1 += f" | K-BB% {sm.away_pitcher.k_bb_pct:.1f}%"
                pitcher_lines.append(p1)
                p2 = f"**{sm.home_pitcher.name}** (HOME): xFIP {sm.home_pitcher.xfip:.2f}"
                if sm.home_pitcher.siera: p2 += f" | SIERA {sm.home_pitcher.siera:.2f}"
                if sm.home_pitcher.k_bb_pct: p2 += f" | K-BB% {sm.home_pitcher.k_bb_pct:.1f}%"
                pitcher_lines.append(p2)
            else:
                pitcher_lines.append(f"Pitchers: {sm.away_pitcher.name} vs {sm.home_pitcher.name}")
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
            for side, team_name, bp in [("away", sm.away_team, sm.away_bullpen), ("home", sm.home_team, sm.home_bullpen)]:
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

            embed.add_field(
                name="📊 Python Win Probabilities (from Sabermetrics)",
                value=(f"🚶 **{sm.away_team}**: {probs['away_prob']*100:.1f}%\n"
                       f"🏠 **{sm.home_team}**: {probs['home_prob']*100:.1f}%"),
                inline=False,
            )
            embed.add_field(name="📐 Calculation Justification", value=probs["justification"], inline=False)

            ev_lines = []
            kelly_lines = []
            positive_ev_outcomes = []

            for outcome, label, emoji in [("away", sm.away_team, "🚶"), ("home", sm.home_team, "🏠")]:
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
                            f"Expected profit: **${kelly['expected_profit']:,.2f}** ({kelly['expected_roi_pct']:+.1f}% ROI)"
                        )
                elif r["ev"] is not None:
                    ev_lines.append(f"❌ {emoji} **{label}**: Price {price_str} | Odds {odds_str} | EV {ev_str}")
                else:
                    ev_lines.append(f"⚪ {emoji} **{label}**: Price {price_str} | No EV data")

            embed.add_field(name="CLOB Prices & EV", value="\n".join(ev_lines) if ev_lines else "No EV data available", inline=False)

            if kelly_lines:
                embed.add_field(name=f"💰 Kelly Stakes ({kelly_label}, Bankroll: ${bankroll:,.2f})", value="\n".join(kelly_lines), inline=False)

            embed.add_field(name="AI Confidence", value=f"{sm.confidence*100:.0f}%", inline=True)

            if positive_ev_outcomes:
                total_stake = sum(
                    calculate_kelly_bet(ev_results[o]["ev"], ev_results[o]["prob"],
                        ev_results[o]["odds"], bankroll, kelly_frac, threshold)["stake"]
                    for o in positive_ev_outcomes
                )
                total_expected = sum(
                    calculate_kelly_bet(ev_results[o]["ev"], ev_results[o]["prob"],
                        ev_results[o]["odds"], bankroll, kelly_frac, threshold)["expected_profit"]
                    for o in positive_ev_outcomes
                )
                total_kelly_pct = (total_stake / bankroll) * 100 if bankroll > 0 else 0
                embed.add_field(name="📋 Bet Summary", value=(
                    f"**{len(positive_ev_outcomes)}** +EV outcome(s)\n"
                    f"Total stake: **${total_stake:,.2f}** ({total_kelly_pct:.1f}% of bankroll)\n"
                    f"Total expected profit: **${total_expected:,.2f}**"
                ), inline=True)
            else:
                embed.add_field(name="📋 Bet Summary", value="No +EV outcomes above threshold — no bets recommended.", inline=True)

            embed.set_footer(text=f"Engine: {SURPLUS_MODEL} via Surplus | Kelly: f* = EV/(odds−1) × {kelly_frac} | Max 25%/bet | Python calc")

            await ctx.send(embed=embed)
            print(f"[MLB CMD] Embed sent successfully", file=sys.stderr)

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
    threshold = get_ev_threshold(gid)
    kelly_frac = get_kelly_fraction(gid)
    bankroll = get_bankroll(gid)
    api_status = "✅ Configured" if SURPLUS_API_KEY else "❌ Not set"
    kelly_label = "Full-Kelly" if kelly_frac == 1.0 else f"{kelly_frac*100:.0f}%-Kelly"

    embed = discord.Embed(
        title="⚾ MLB Bot Status",
        description="\n".join([
            f"**Mode:** MLB Sabermetric + Kelly Criterion",
            f"**EV Threshold:** {threshold*100:.1f}%",
            f"**Kelly Fraction:** {kelly_frac} ({kelly_label})",
            f"**Bankroll:** ${bankroll:,.2f}",
            f"**AI Model:** {SURPLUS_MODEL}",
            f"**Surplus API:** {api_status}",
            f"**Architecture:** LLM → Sabermetrics → Python × Kelly = EV",
        ]),
        color=discord.Color.green() if SURPLUS_API_KEY else discord.Color.orange(),
    )
    await ctx.send(embed=embed)


@bot.event
async def on_ready():
    print(f"[MLB Bot] ========================================", file=sys.stderr)
    print(f"[MLB Bot] Online: {bot.user.name} ({bot.user.id})", file=sys.stderr)
    print(f"[MLB Bot] Mode: Sabermetric → Python Probabilities → Kelly", file=sys.stderr)
    print(f"[MLB Bot] Model: {SURPLUS_MODEL}", file=sys.stderr)
    print(f"[MLB Bot] EV Threshold: {DEFAULT_EV_THRESHOLD*100:.1f}%", file=sys.stderr)
    print(f"[MLB Bot] Kelly Fraction: {DEFAULT_KELLY_FRACTION}", file=sys.stderr)
    print(f"[MLB Bot] Bankroll: ${DEFAULT_BANKROLL:,.2f}", file=sys.stderr)
    print(f"[MLB Bot] API Key Set: {'Yes' if SURPLUS_API_KEY else 'NO — !mlb will fail'}", file=sys.stderr)
    print(f"[MLB Bot] ========================================", file=sys.stderr)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    print(f"[MLB Bot Error] Command '{ctx.command}': {type(error).__name__}: {error}", file=sys.stderr)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    try:
        await ctx.send(f"❌ **Command Error**: {type(error).__name__}: {error}")
    except Exception:
        pass


async def main():
    if not DISCORD_TOKEN:
        print("FATAL: MLB_BOT_TOKEN or DISCORD_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    print(f"[MLB Bot] Starting with token: {DISCORD_TOKEN[:8]}...", file=sys.stderr)
    async with bot:
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown():
        print("[MLB Bot] Shutting down...", file=sys.stderr)
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
        print("[MLB Bot] Stopped cleanly", file=sys.stderr)