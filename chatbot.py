"""
chatbot.py — General AI Agent (ChatGPT-style) via Surplus API
==============================================================
Simple Discord bot: !ask <anything> → Surplus API → response.
Same model + API key as EV bot / MLB bot, but no domain logic.
Just a raw LLM interface with conversation context (last 20 msgs).

Usage:
  1. Create a Discord bot at https://discord.com/developers/applications
  2. Add CHATBOT_TOKEN to your .env file
  3. Invite bot to your server
  4. systemctl enable --now chatbot
"""

import os, sys, asyncio, signal, traceback
from openai import AsyncOpenAI
import discord
from discord.ext import commands

# ── env config ──────────────────────────────────────────────
DISCORD_TOKEN   = os.getenv("CHATBOT_TOKEN")
SURPLUS_API_KEY = os.getenv("SURPLUS_API_KEY")
SURPLUS_BASE_URL = os.getenv("SURPLUS_API_URL", "https://api.surplusintelligence.ai/min30/v1")
SURPLUS_MODEL   = os.getenv("SURPLUS_MODEL", "gpt-5.4")

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant running on the Surplus API. "
    "Answer questions concisely and accurately. Use bullet points for lists. "
    "Keep responses skimmable. No markdown code blocks unless the user asks for code. "
    "If you don't know something, say so — don't guess."
)

# ── OpenAI ──────────────────────────────────────────────────

def get_client() -> AsyncOpenAI:
    if not SURPLUS_API_KEY:
        raise RuntimeError("SURPLUS_API_KEY not set")
    return AsyncOpenAI(api_key=SURPLUS_API_KEY, base_url=SURPLUS_BASE_URL)


async def chat(messages: list) -> str:
    """Send conversation to Surplus API, return response text."""
    client = get_client()
    response = await client.chat.completions.create(
        model=SURPLUS_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=4000,
    )
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content or "(empty response)"
    return "(no response)"


# ── Discord Bot ─────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Per-channel conversation history (last 20 messages per channel)
conversations: dict[int, list[dict]] = {}
MAX_HISTORY = 20


@bot.command(name="ask")
async def cmd_ask(ctx, *, question: str = ""):
    """Ask the AI anything. !ask <your question>"""
    if not question.strip():
        await ctx.send("Usage: `!ask <your question>`\nExample: `!ask what is xFIP in baseball?`")
        return

    channel_id = ctx.channel.id

    # Build conversation history
    if channel_id not in conversations:
        conversations[channel_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversations[channel_id].append({"role": "user", "content": question})

    # Trim to last N messages + system prompt
    if len(conversations[channel_id]) > MAX_HISTORY + 1:
        conversations[channel_id] = [conversations[channel_id][0]] + conversations[channel_id][-(MAX_HISTORY):]

    async with ctx.typing():
        try:
            reply = await chat(conversations[channel_id])
            conversations[channel_id].append({"role": "assistant", "content": reply})

            # Discord has a 2000 char limit per message — split if needed
            if len(reply) <= 2000:
                await ctx.send(reply)
            else:
                chunks = [reply[i:i+2000] for i in range(0, len(reply), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
        except Exception as e:
            await ctx.send(f"❌ **API Error**: {type(e).__name__}: {e}")
            # Remove the user message that failed so it doesn't pollute history
            if conversations[channel_id] and conversations[channel_id][-1]["role"] == "user":
                conversations[channel_id].pop()


@bot.command(name="clear")
async def cmd_clear(ctx):
    """Clear conversation history for this channel."""
    channel_id = ctx.channel.id
    conversations.pop(channel_id, None)
    await ctx.send("✅ Conversation history cleared.")


@bot.command(name="chatmodel")
async def cmd_model(ctx):
    """Show current AI model."""
    await ctx.send(f"**Model:** `{SURPLUS_MODEL}`\n**API:** `{SURPLUS_BASE_URL}`")


@bot.command(name="chathelp")
async def cmd_help(ctx):
    await ctx.send(
        "**AI Chatbot Commands**\n"
        "`!ask <question>` — Ask the AI anything\n"
        "`!clear` — Clear conversation history\n"
        "`!chatmodel` — Show current model\n"
        "`!chathelp` — Show this help\n\n"
        f"**Model:** `{SURPLUS_MODEL}` | **Context:** last {MAX_HISTORY} messages per channel"
    )


@bot.event
async def on_ready():
    print(f"[Chatbot] Online: {bot.user.name} ({bot.user.id}) | Model: {SURPLUS_MODEL}", file=sys.stderr)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    print(f"[Chatbot Error] {type(error).__name__}: {error}", file=sys.stderr)
    try:
        await ctx.send(f"❌ **Error**: {type(error).__name__}: {error}")
    except Exception:
        pass


async def main():
    if not DISCORD_TOKEN:
        print("FATAL: CHATBOT_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)
    if not SURPLUS_API_KEY:
        print("FATAL: SURPLUS_API_KEY not set in .env", file=sys.stderr)
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
        print("[Chatbot] Stopped cleanly", file=sys.stderr)