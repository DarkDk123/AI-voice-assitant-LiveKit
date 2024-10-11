"""
server.py
---

LiveKit backend server that creates rooms, connects to the clients through LiveKit Cloud.

"""

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero, openai, deepgram
import os

# Load .env
from dotenv import load_dotenv

load_dotenv()


async def entry_point(ctx: JobContext):
    # Intial Context
    init_llm_context = llm.ChatContext().append(
        role="system",
        text=(
            # It will be concatenated!
            "Your are a Rajneesh Osho, an indian philosopher. Your Interface with users will be voice!",
            "you should use short, concise responses, and avoid usage of unpronounceable punctuations.",
        ),
    )

    # Connecting to the room!
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = (
        VoiceAssistant(
            vad=silero.VAD.load(),
            stt=openai.STT(
                base_url="https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
                api_key=os.environ["HUGGING_FACE_API_TOKEN"],
            ),
            llm="",
            tts=openai.TTS(

                base_url="https://api-inference.huggingface.co/models/microsoft/speecht5_tts",
                api_key=os.environ["HUGGING_FACE_API_TOKEN"],
            ),
            chat_ctx=init_llm_context,
            allow_interruptions=True,
        ),
    )

    assistant.start(ctx.room)

    await assistant.say("Hey, how can i help you today??")


# Runner code
if __name__ == "__main__":
    # Running the app.
    cli.run_app(WorkerOptions(entrypoint_fnc=entry_point))
