from google import genai
from pydantic import BaseModel, Field
from typing import Literal
import json_stream


class Feedback(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str = Field(description="Short summary in 10 words")
    thinking: str = Field(description="Steps taken to produce the sentiment result")


client = genai.Client()
prompt = (
    "The new UI is incredibly intuitive and visually appealing. Great job. Add a very long summary to test streaming!"
)

response_stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": Feedback.model_json_schema(),
    },
)

text_chunks = (chunk.candidates[0].content.parts[0].text for chunk in response_stream)
# data = json_stream.load(text_chunks, persistent=True)
data = json_stream.load(text_chunks)
# get the data without waiting for whole document to be received!
print(data["sentiment"])
print("-----------------------")
print(data["summary"])
print("-----------------------")
print(data["thinking"])

# will yield an error! stream is exhausted
# print(data["sentiment"])
