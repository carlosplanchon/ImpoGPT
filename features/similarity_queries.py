#!/usr/bin/env python3

import pandas as pd

from langchain.embeddings.openai import OpenAIEmbeddings
import os

import pinecone

from typing import Any, Optional, Dict, List

from pinecone import QueryResponse

import string

import json


def get_section_key(section_key: str) -> str:
    if section_key in ["Artículo Unico", "Resumen", "Summary"]:
        return "1"

    section_digits: str = "".join([l for l in section_key if l in string.digits])
    return section_digits


###################
#   CREDENTIALS   #
###################
# os.environ["OPENAI_API_KEY"]
# os.environ["PINECONE_API_KEY"]

#############
#   CONST   #
#############
PINECONE_ENVIROMENT: str = "us-east1-gcp"
PINECONE_INDEX_NAME: str = "impolaws"

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=PINECONE_ENVIROMENT
)
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)


#############
#   FILES   #
#############
with open("static/impodata.json", "r") as f:
    impo_laws = json.load(f)


def compose_filter(
    law_filter: Optional[List[str]],
    section_filter: Optional[List[str]]
        ) -> Dict[str, Any]:
    filter_dict: Dict[str, Any] = {}

    if law_filter is not None:
        filter_dict["law_id"] = {
            "$in": law_filter
        }
    
    if section_filter is not None:
        filter_dict["section_id"] = {
            "$in": section_filter
        }

    return filter_dict


def convert_law_result(law_result: dict[str, str]) -> dict[str, str]:
    converted_law_result: dict[str, str] = {}

    if "law_title" in law_result:
        del law_result["law_title"]

    for k, v in law_result.items():
        section_key = get_section_key(section_key=k)
        converted_law_result[section_key] = v

    return converted_law_result


def similarity_query(
    api_key: str,
    text: str,
    law_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
    top_k: Optional[int] = 1
        ) -> List[QueryResponse]:
    ada_embeddings_engine = OpenAIEmbeddings(
        openai_api_key=api_key
    )
    # Request embeddings for text:
    query_result = ada_embeddings_engine.embed_query(text)
    filter_dict = compose_filter(
        law_filter=law_filter,
        section_filter=section_filter
    )
    pinecone_result = pinecone_index.query(
        vector=query_result,
        filter=filter_dict,
        top_k=top_k,
        include_metadata=True,
        namespace="laws"
    )

    matches = pinecone_result["matches"]

    results = []
    for m in matches:
        converted_m = m.to_dict()
        del converted_m["values"]

        law_id: str = converted_m["metadata"]["law_id"]
        section_id: str = converted_m["metadata"]["section_id"]
        law_result = impo_laws[law_id].copy()
        law_title: str = law_result["law_title"]
        print(f"Law ID: {law_id}")
        print(f"Section ID: {section_id}")

        if section_id == "Summary":
            section_id = "1"
        law_result: dict[str, str] = convert_law_result(
            law_result=law_result)

        # print("--- LAW FOUND ---")
        # print(law_result)
        matched_section: str = law_result[section_id]

        converted_m["law_title"] = law_title
        converted_m["matched_section"] = matched_section

        results.append(converted_m)
    # Map from first result to small_embedings_df id:
    return results
