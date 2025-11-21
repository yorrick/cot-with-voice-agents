from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, Field
from typing import Literal
import json_stream


class Feedback(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str = Field(description="Short summary in 10 words")
    thinking: str = Field(description="Steps taken to produce the sentiment result")


client = OpenAI()
prompt = (
    "The new UI is incredibly intuitive and visually appealing. Great job. Add a very long summary to test streaming!"
)


# 1. Define a Pydantic model for the tool’s input:
class GenerateSummaryArgs(BaseModel):
    text: str


# 2. Use pydantic_function_tool() to wrap the model:
tools = [
    pydantic_function_tool(
        GenerateSummaryArgs,  # <-- Pass the class, NOT a function
        name="generate_summary",
        description="Generate a summary of a text",
    )
]


with client.responses.stream(
    model="gpt-5.1",
    tools=tools,
    input=[
        {"role": "user", "content": prompt},
    ],
    text_format=Feedback,
) as response_stream:
    text_chunks = (event.delta for event in response_stream if event.type == "response.output_text.delta")
    # # data = json_stream.load(text_chunks, persistent=True)
    data = json_stream.load(text_chunks)
    # get the data without waiting for whole document to be received!
    print(data["sentiment"])
    print("-----------------------")
    print(data["summary"])
    print("-----------------------")
    print(data["thinking"])

    # # will yield an error! stream is exhausted
    # # print(data["sentiment"])
