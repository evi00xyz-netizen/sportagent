import os
import json
import asyncio
import aiohttp
import discord
from discord.ext import commands
import openai

# configuration
DEFAULT_MIN_EDGE = 0.05  # 5% default minimum edge required

# discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# store user settings (min_edge)
user_settings = {}

def get_min_edge(guild_id: int) -> float:
    return user_settings.get(guild_id, {}).get("min_edge", DEFAULT_MIN_EDGE)

def set_min_edge(guild_id: int, edge: float):
    if guild_id not in user_settings:
        user_settings[guild_id] = {}
    user_settings[guild_id]["min_edge"] = edge

async def fetch_from_custom_api(match_query: str, api_url: str, api_key: str = None) -> dict:
    """
    queries your custom intelligence API for match probability matrix data.
    strictly avoids passing market odds into the model context.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    payload = {
        "match": match_query,
        "mode": "probability_matrix",
        "exclude_odds": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=payload, headers=headers, timeout=30) as resp:
            if resp.status != 200:
                raise Exception(f"Custom API returned status {resp.status}: {await resp.text()}")
            data = await resp.json()
            return data

async def fetch_from_openai(match_query: str) -> dict:
    """
    fallback: queries openai gpt-4o / web search to calculate true win probabilities.
    strictly excludes bookmaker odds from the prompt context.
    """
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    You are an expert quantitative sports analyst and match probability matrix engine.
    Analyze the following upcoming match: "{match_query}".

    CRITICAL RULE: STRICTLY DO NOT search for, mention, or use any betting odds, bookmaker lines, or market prices.
    Your job is to estimate the PURE TRUE WIN/DRAW/LOSS probabilities based solely on fundamental football/sports data.

    Perform a match-level Probability Matrix analysis considering:
    1. Team Form & Recent Performance (last 6-10 matches, xG trends)
    2. Head-to-Head History (venue-adjusted)
    3. Squad News (Injuries, Suspensions, Key Absences)
    4. Tactical Matchup & Style of Play (Possession, Pressing, Set Pieces)
    5. Home Advantage / Venue Factors
    6. Fatigue / Rest Days & Travel Distance

    Return your output strictly as a JSON object with this exact structure:
    {{
      "match_name": "Team A vs Team B",
      "home_team": "Team A",
      "away_team": "Team B",
      "true_probabilities": {{
        "home_win": 0.45,
        "draw": 0.28,
        "away_win": 0.27
      }},
      "matrix_factors": {{
        "home_form_rating": "8/10",
        "away_form_rating": "6/10",
        "key_absences_home": "Player X (Injured)",
        "key_absences_away": "None",
        "tactical_edge": "Home team high-press favors transition against Away defense"
      }},
      "forecast_result": "Home Win (2-1 or 2-0 expected scoreline)",
      "confidence_score": 0.75
    }}
    Ensure true_probabilities sum up to 1.0.
    """

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quantitative sports probability matrix engine. Respond ONLY in valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    content = response.choices[0].message.content
    return json.loads(content)

async def fetch_true_probabilities(match_query: str) -> dict:
    """
    routes match probability retrieval to custom API if configured via environment,
    otherwise falls back to OpenAI GPT-4o.
    """
    custom_api_url = os.getenv("SURPLUS_API_URL")
    custom_api_key = os.getenv("SURPLUS_API_KEY")

    if custom_api_url:
        return await fetch_from_custom_api(match_query, custom_api_url, custom_api_key)
    else:
        return await fetch_from_openai(match_query)

def calculate_edges(true_probs: dict, odds_home: float = None, odds_draw: float = None, odds_away: float = None):
    """
    calculates edge against market odds if provided.
    devigs implied probabilities and computes edge = true_prob - implied_prob.
    """
    if not odds_home or not odds_away:
        return None

    # raw implied probabilities
    raw_implied_h = 1.0 / odds_home
    raw_implied_d = (1.0 / odds_draw) if odds_draw else 0.0
    raw_implied_a = 1.0 / odds_away

    total_implied = raw_implied_h + raw_implied_d + raw_implied_a

    # de-vigged implied probabilities
    implied_h = raw_implied_h / total_implied
    implied_d = raw_implied_d / total_implied if odds_draw else 0.0
    implied_a = raw_implied_a / total_implied

    true_h = true_probs["home_win"]
    true_d = true_probs.get("draw", 0.0)
    true_a = true_probs["away_win"]

    edge_h = true_h - implied_h
    edge_d = true_d - implied_d if odds_draw else None
    edge_a = true_a - implied_a

    return {
        "implied": {"home": implied_h, "draw": implied_d, "away": implied_a},
        "edges": {"home": edge_h, "draw": edge_d, "away": edge_a},
        "overround_margin": (total_implied - 1.0) * 100
    }

@bot.command(name="minedge")
async def cmd_set_min_edge(ctx, edge_pct: float):
    """set min edge threshold (e.g. !minedge 0.05 for 5%)"""
    guild_id = ctx.guild.id if ctx.guild else ctx.author.id
    set_min_edge(guild_id, edge_pct)
    await ctx.send(f"min edge threshold updated to {edge_pct * 100:.1f}% ({edge_pct:.3f})")

@bot.command(name="match")
async def cmd_match(ctx, *, args: str):
    """
    calculate true match probabilities and check edge.
    usage:
    !match Arsenal vs Chelsea
    !match Arsenal vs Chelsea | odds: 2.10, 3.40, 3.20
    """
    await ctx.trigger_typing()

    match_query = args
    odds_home = odds_draw = odds_away = None

    if "|" in args:
        parts = args.split("|")
        match_query = parts[0].strip()
        odds_str = parts[1].replace("odds:", "").strip()
        try:
            odds_vals = [float(x.strip()) for x in odds_str.split(",")]
            if len(odds_vals) == 3:
                odds_home, odds_draw, odds_away = odds_vals
            elif len(odds_vals) == 2:
                odds_home, odds_away = odds_vals
        except ValueError:
            await ctx.send("invalid odds format. use: !match Team A vs Team B | odds: 2.10, 3.40, 3.20")
            return

    guild_id = ctx.guild.id if ctx.guild else ctx.author.id
    current_min_edge = get_min_edge(guild_id)

    try:
        data = await fetch_true_probabilities(match_query)
    except Exception as e:
        await ctx.send(f"error calculating probabilities: {str(e)}")
        return

    true_p = data["true_probabilities"]
    factors = data.get("matrix_factors", {})
    forecast = data.get("forecast_result", "N/A")

    source_label = "Custom Surplus Intelligence API" if os.getenv("SURPLUS_API_URL") else "OpenAI Matrix Model"

    embed = discord.Embed(
        title=f"Match Probability Matrix: {data.get('match_name', match_query)}",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="True Win Probabilities (No Odds Bias)",
        value=f"**{data.get('home_team', 'Home')}**: {true_p['home_win']*100:.1f}%\n"
              f"**Draw**: {true_p.get('draw', 0)*100:.1f}%\n"
              f"**{data.get('away_team', 'Away')}**: {true_p['away_win']*100:.1f}%",
        inline=False
    )

    embed.add_field(name="Predicted Match Result", value=forecast, inline=False)

    if factors:
        factor_text = "\n".join([f"• **{k.replace('_', ' ').title()}**: {v}" for k, v in factors.items()])
        embed.add_field(name="Probability Matrix Key Factors", value=factor_text, inline=False)

    if odds_home and odds_away:
        calc = calculate_edges(true_p, odds_home, odds_draw, odds_away)
        edges = calc["edges"]
        implied = calc["implied"]

        edge_msg = (
            f"**Home Implied**: {implied['home']*100:.1f}% | **Edge**: {edges['home']*100:+.1f}%\n"
        )
        if odds_draw:
            edge_msg += f"**Draw Implied**: {implied['draw']*100:.1f}% | **Edge**: {edges['draw']*100:+.1f}%\n"
        edge_msg += f"**Away Implied**: {implied['away']*100:.1f}% | **Edge**: {edges['away']*100:+.1f}%\n"

        embed.add_field(name="Market Edge Analysis", value=edge_msg, inline=False)

        # value bet detection
        value_bets = []
        if edges["home"] >= current_min_edge:
            value_bets.append(f"Home ({data.get('home_team')}) Edge: {edges['home']*100:+.1f}%")
        if edges.get("draw") and edges["draw"] >= current_min_edge:
            value_bets.append(f"Draw Edge: {edges['draw']*100:+.1f}%")
        if edges["away"] >= current_min_edge:
            value_bets.append(f"Away ({data.get('away_team')}) Edge: {edges['away']*100:+.1f}%")

        if value_bets:
            embed.add_field(
                name=f"VALUE BET RECOMMENDED (Min Edge >= {current_min_edge*100:.1f}%)",
                value="\n".join([f"✅ {vb}" for vb in value_bets]),
                inline=False
            )
        else:
            embed.add_field(
                name=f"No Value Bet (Min Edge Threshold = {current_min_edge*100:.1f}%)",
                value="❌ Market implied odds match or exceed true probabilities. Pass.",
                inline=False
            )

    embed.set_footer(text=f"Engine: {source_label} | Min Edge: {current_min_edge*100:.1f}%")
    await ctx.send(embed=embed)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} ({bot.user.id})")

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Error: DISCORD_BOT_TOKEN environment variable not set.")
    else:
        bot.run(token)
