#!/usr/bin/env python3

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


def get_chatgpt_response(api_key: str, prompt: str):
    chat = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key
    )
    resp = chat.generate(
        messages=[[HumanMessage(content=prompt)]]
    )
    content = resp.generations[0][0].text
    return content
