import logging
import os
import re
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
import random
load_dotenv()

logger = logging.getLogger("demo")
logger.setLevel(logging.INFO)


class AssistantFnc(llm.FunctionContext):
    """
    The class defines a set of LLM functions that the assistant can execute.
    """

    @llm.ai_callable()
    async def query_information(
        self,
        query: Annotated[
            str, llm.TypeInfo(description="User query for retrieving information related to get perfume Information from the retrieval system")
        ],
    ):
        """
        Searches and retrieves information Perfumes and Sunna Musk DB Information based on the provided query.
        """
        logger.debug(f"query_information function called with query: {query}")
        call_ctx = AgentCallContext.get_current()

        # List of possible messages to send
        messages = [
            f"Searching for information related to: '{query}'. This may take a moment."
        ]
        
        # Send the message to the user
        await call_ctx.agent.say(messages[0], add_to_chat_ctx=True)

        headers = {
            "Content-Type": "application/json",
        }
        
        url = "http://136.243.132.228:9061/query_agent_rag_sunna"
        payload = {
            "query_text": query
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        
                        if "response" in response_data:
                            return response_data["response"]
                        else:
                            return "The response format from the system was unexpected."
                    else:
                        return f"Failed to query the system, status code: {response.status}, reason: {response.reason}"
        except Exception as e:
            logger.error("query_information function encountered an error: %s", e)
            return f"I'm sorry, I encountered an error while querying the system: {str(e)}"




def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance
    initial_chat_ctx = llm.ChatContext().append(
        text=(
            "You are helpfull conversational assistant at Sunna Musk. Sunna Musk is a London Based Perfume Store. You will conversation with user in polite manner. "
            "please call the function  user want to get additional information or you need it for specific answer. Call the function if you need to get answer of specific question or get information about perfumes."
            "Sunnamusk collaborates with highly skilled perfumers to produce an exclusive range of the finest fragrances. Our passion and commitment to quality craftsmanship resonates with discerning customers who appreciate exquisite scents. Since our conception, we have retained a disruptive and agile nature that sets us apart from brand competitors - we refuse to match the status quo of perfumery."
        ),
        role="system",
    )
    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
    )


    @agent.on("function_calls_collected")
    def on_function_calls_collected(fnc_calls):
        logger.info(f"function calls collected: {fnc_calls}")

    @agent.on("function_calls_finished")
    def on_function_calls_finished(fnc_calls):
        logger.info(f"function calls finished: {fnc_calls}")

    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)

    await agent.say("Hello, Welcome to Sunna Musk. I am Meena, How I can assist you?")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )