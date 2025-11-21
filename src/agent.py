import json
import logging
from collections.abc import AsyncIterable

import ijson
from dotenv import load_dotenv
from google.protobuf import text_format
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    FlushSentinel,
    JobContext,
    JobProcess,
    ModelSettings,
    RunContext,
    cli,
    function_tool,
    inference,
    llm,
    room_io,
)
from livekit.agents.llm import FunctionTool, RawFunctionTool
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pydantic import BaseModel, Field

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class ResponseAndThinking(BaseModel):
    response: str = Field(description="Agent response")
    thinking: str = Field(description="Steps taken to produce the response")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM), (
            "llm_node should only be used with LLM (non-multimodal/realtime APIs) nodes"
        )

        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        activity_llm = activity.llm

        conn_options = activity.session.conn_options.llm_conn_options
        async with activity_llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            conn_options=conn_options,
            response_format=ResponseAndThinking,
        ) as stream:
            last_response = ""
            thinking_value = ""

            # Create a target coroutine that receives events
            def event_collector():
                nonlocal last_response, thinking_value
                while True:
                    prefix, event, value = yield
                    if prefix == "response" and event == "string":
                        current_response = value
                        if len(current_response) > len(last_response):
                            last_response = current_response
                    elif prefix == "thinking" and event == "string":
                        thinking_value = value

            # Create the push parser with our event collector
            target = event_collector()
            next(target)  # Prime the coroutine
            parser = ijson.parse_coro(target)

            async for chunk in stream:
                if isinstance(chunk, llm.ChatChunk) and chunk.delta:
                    # If there are tool calls, yield the chunk as-is
                    if chunk.delta.tool_calls:
                        yield chunk
                    elif chunk.delta.content:
                        # Get current response length before sending
                        prev_len = len(last_response)

                        # Send chunk to parser
                        parser.send(chunk.delta.content.encode("utf-8"))

                        # Yield any new content
                        if len(last_response) > prev_len:
                            yield last_response[prev_len:]

            # Close the parser (may have incomplete JSON if tool was called)
            try:
                parser.close()
            except ijson.IncompleteJSONError:
                # Expected when stream ends due to tool call
                pass

            # Log thinking
            if thinking_value:
                logger.info(f"Chain of thought: {thinking_value}")


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="elevenlabs/eleven_turbo_v2_5"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    @session.on("function_tools_executed")
    def on_function_tools_executed(event):
        """Log tool calls after they are executed."""
        for call, output in event.zipped():
            logger.info(
                f"Tool executed: {call.name} with args: {call.arguments} -> "
                f"{'error' if output.is_error else 'success'}: {output.output}"
            )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
