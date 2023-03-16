#!/usr/bin/env python3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path

from pydantic import BaseModel

from features import chatgpt
from features.querier import Querier
from features import similarity_queries

import pinecone
from langchain.cache import InMemoryCache

import os
import langchain
import sys

from typing import Any, List, Optional


if pinecone_api_key := os.environ.get("PINECONE_API_KEY"):
    pinecone.init(
        api_key=pinecone_api_key,
        environment=os.environ.get(
            "PINECONE_ENVIRONMENT",
            os.environ.get("PINECONE_ENVIROMENT")
        )
    )

langchain.llm_cache = InMemoryCache()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)


@app.get("/")
def demo_get():
    return FileResponse("static/index.html")


@app.get("/python_version")
def python_version():
    """
    Returns the current Python version of the project.
    """
    return sys.version


@app.get("/healthcheck")
def healthcheck():
    return {"status": "alive"}


class Prompt(BaseModel):
    api_key: str
    prompt: str


@app.post("/send_prompt")
def send_prompt(prompt: Prompt):
    result = chatgpt.get_chatgpt_response(
        api_key=prompt.api_key,
        prompt=prompt.prompt
    )
    return {"result": result}



class SimilarityRequest(BaseModel):
    api_key: str
    prompt: str
    # Filter by law code.
    law_filter: Optional[List[str]]
    top_k: Optional[int] = 10


@app.post("/similarity_search")
def send_similarity_search(similarity: SimilarityRequest):
    """
    Queries Pinecone using the embeddings of the input text, retrieves the closest match, and
    returns it as a JSON payload.

     **Body:**\n
    `api_key`: OpenAI API key.\n
    `prompt`: The text to search for similarities.\n
    `law_filter` (optional): A filter to limit the search to specific law codes.\n
    `top_k` (optional): The number of sections that will be returned.\n
    **Returns:** A JSON payload containing the fields Law, Section, Title and text.
    """
    if similarity.top_k > 20:
        return {"error": "top_k cannot be greater than 10."}

    laws = similarity_queries.similarity_query(
        api_key=similarity.api_key,
        text=similarity.prompt,
        top_k=similarity.top_k,
        law_filter=similarity.law_filter
    )
    return {"result": laws}


class Question(BaseModel):
    api_key: str
    prompt: str
    law_filter: Optional[List[str]] = None


@app.post("/send_question")
def main_query(question: Question) -> dict[str, Any]:
    """
    Sends a question to be answered by the sub-question process.\n
    **Body:**\n
    `api_key`: OpenAI API key.\n
    `prompt:` The legal question to be answered. Example: "¿Cuando se abolió la pena de muerte en Uruguay?")\n
    `law_filter`: Limit the process to only look into the selected laws. Example: ["3238-1907", "5350-1915"]\n
    """
    querier_obj = Querier(
        openai_api_key=question.api_key
    )
    resp = querier_obj.main_query(
        query=question.prompt,
        n=3,
        filter_by_law=question.law_filter
    )
    return resp
