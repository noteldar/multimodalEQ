from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.llm import LLMStream
from .log import logger
import asyncio
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.types import StreamLanguage
import os
from hume.expression_measurement.stream.socket_client import StreamConnectOptions


HUME_API_KEY = os.getenv("HUME_API_KEY")


async def hume_ev_text(user_text):
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    model_config = Config(language=StreamLanguage())
    stream_options = StreamConnectOptions(config=model_config)
    async with client.expression_measurement.stream.connect(
        options=stream_options
    ) as socket:
        text_emotions = await socket.send_text(user_text)
    return text_emotions


async def hume_ev_voice(fileb64):
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    model_config = Config(prosody={})
    stream_options = StreamConnectOptions(config=model_config)
    async with client.expression_measurement.stream.connect(
        options=stream_options
    ) as socket:
        voice_prosody = await socket.send_file(fileb64)
    return voice_prosody
