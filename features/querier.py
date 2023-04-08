#!/usr/bin/env python3

from typing import TypedDict
import re

import hashlib
import pandas as pd
import pinecone
from langchain import (
    BasePromptTemplate,
    FewShotPromptTemplate,
    LLMChain,
    PromptTemplate,
)
# from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from openai.error import RateLimitError
from retry import retry
from features.mongo import MongoQuerier
from features.chain import Chain
import os

from features.similarity_queries import similarity_query
from features import chatgpt

from langchain.chat_models import ChatOpenAI

from features import templates

PROMPT_MAX_TOKENS: int = 4080


class Querier(Chain):
    """
    Queriers are responsible for answering questions about the dataset,
    using the pre-populated model.

    The high level flow is as follows:
    1. Determine a set of sub-questions necessary to answer the original.
    2. For each sub-question, determine a set of queries that would render relevant facts.
    3. For each query, search the vector store for similar_sections _or_ queries that are similar to the embedded query.
    4. Extract the similar_sections from these results.
    5. Recursively summarize up the tree until the original question is answered.
    """

    def __init__(
        self,
        openai_api_key: str,
        debug: bool = False
            ):
        super().__init__()

        self.openai_api_key = openai_api_key

        self.instance_external_resources()

        self.THOUGHT_PROCESS = {}

        self.prompts_sent = []

        self.mongo_querier = MongoQuerier()

    def instance_external_resources(self):
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key,
        )

        self.openai_embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key)

    # Questions:
    @retry(
        exceptions=RateLimitError,
        tries=5,
        delay=10,
        backoff=2,
        max_delay=120,
        jitter=(0, 10),
    )
    def query_llm(
        self,
        prompt: BasePromptTemplate,
        initial: str = "",
        prefix: str = "",
        quiet: bool = False,
        pltags: list[str] | None = None,
        tag: str | None = "no_tag",
        **kwargs,
    ) -> list[str] | str:

        query_save_object = {"prompt": prompt.format_prompt(**kwargs).dict()["text"]}

        input_tokens_amt: int = self.llm.get_num_tokens(
            text=query_save_object["prompt"])

        self.llm.max_tokens = PROMPT_MAX_TOKENS - input_tokens_amt

        chain = LLMChain(llm=self.llm, prompt=prompt)
        results = initial + chain.run(**kwargs)

        if initial and prefix:
            results = self._parse(
                results=results.splitlines(),
                prefix=prefix
            )

        query_save_object["response"] = results
        query_save_object["tag"] = tag
        query_save_object["question"] = kwargs.get("query")
        self.prompts_sent.append(query_save_object)
        return results

    def query_similar_sections(
            self,
            query: str,
            n: int,
            filter_by_law: list[str] | None,
    ):
        """ Make an embedding and query against Pinecone. """
        print("> Query similar sections.:")
        print(query)

        result = similarity_query(
            api_key=self.openai_api_key,
            text=query,
            law_filter=filter_by_law,
            top_k=n
        )

        metadata = [row["metadata"] for row in result]

        self.THOUGHT_PROCESS["sources"] = metadata

        sections_matched = [row["matched_section"] for row in result]
        uniq_sections_matched: list[str] = list(set(sections_matched))
        # print("#" * 50)
        # print("> UNIQ SECTIONS MATCHED:")
        # print(uniq_sections_matched)

        return uniq_sections_matched

    def summarize_similar_sections(
            self,
            similar_sections: list[str],
            query: str
    ):
        # Experimental value.
        threshold: int = 12000
        if (len("".join(similar_sections)) + len(query)) > threshold:
            docs = [
                Document(
                    page_content=section
                ) for section in similar_sections
            ]
            return self.chatgpt_chain_query(
                docs=similar_sections,
                query=query
            )
        else:
            return self.summarize_small_similar_sections(
                similar_sections=similar_sections,
                query=query
            )

    def summarize_small_similar_sections(
            self,
            similar_sections,
            query
    ) -> str:
        bulleted_similar_sections: str = "".join(
            [f"-{section}\n" for section in similar_sections]
        )

        summarize_template: PromptTemplate = \
            templates.summarize_similar_sections_template()

        return self.query_llm(
            prompt=summarize_template,
            query=query,
            sections=bulleted_similar_sections,
            tag="summarize_similar_sections_fast"
        )

    def chatgpt_chain_query(self, docs, query):
        docs_amount = len(docs)
        question_prompt, refine_prompt = \
            templates.summarize_big_similar_sections_templates()

        first_document = docs[0]
        current_response = self.query_llm(
            prompt=question_prompt,
            text=first_document,
            query=query,
            tag="summarize_similar_sections_slow"
        )
        if docs_amount == 1:
            return current_response

        for i in range(1, docs_amount):
            current_response = self.query_llm(
                prompt=refine_prompt,
                existing_answer=current_response,
                text=docs[i],
                query=query,
                tag="summarize_similar_sections_slow"
            )

        return current_response

    def answer_question(
            self,
            question: str,
            n: int,
            filter_by_law,
    ) -> templates.Answer:
        print(f"Ask: {question}")
        print("> Answer question.")
        print(f"FILTER BY LAW: {filter_by_law}")
        print(question)
        similar_sections: list[str] = self.query_similar_sections(
            query=question,
            n=n,
            filter_by_law=filter_by_law,
        )

        print("SUMMARIZE SIMILAR_SECTIONS:")
        answer = self.summarize_similar_sections(
            query=question,
            similar_sections=similar_sections
        )
        print(f"Answer: {answer}")
        return {"question": question, "answer": answer}

    def conclude_step(
            self,
            step: str,
            query: str,
            n: int,
            filter_by_law: list[str] | None
    ) -> templates.Conclusion:
        print(f"Action: {step}")
        QUERIES_TEMPLATE: PromptTemplate = templates.queries_template()
        questions_list = self.query_llm(
            prompt=QUERIES_TEMPLATE,
            initial="-",
            prefix=r"\-",
            query=query,
            step=step,
            n=n,
            tag="subquestions_queries"
        )

        answers: list[templates.Answer] = self._pmap(
            self.answer_question,
            questions_list,
            n,
            filter_by_law,
        )

        ANSWERS_TEMPLATE: FewShotPromptTemplate = \
            templates.answers_template(answers=answers)

        conclusion = self.query_llm(
            prompt=ANSWERS_TEMPLATE,
            query=query,
            step=step,
            tag="answer_subquestion"
        )
        return {"step": step, "conclusion": conclusion}

    def subquestions_conclusion(
            self,
            query: str,
            n: int = 3,
            filter_by_law: list[str] | None = None
    ):
        print(f"Research")

        STEPS_TEMPLATE: PromptTemplate = templates.steps_template()

        # Already tested. A set of 3 steps to answer the question.  
        steps: list[str] = self.query_llm(
            prompt=STEPS_TEMPLATE,
            initial="1.",
            prefix=r"\d+(?:\.)",
            query=query,
            n=n,
            tag="get_subquestions"
        )

        print("--- STEPS ---")
        steps_to_save: list[str] = [str(s) for s in steps]
        for s in steps:
            print(f"> {s}")

        self.THOUGHT_PROCESS["steps"] = steps

        print("--- END STEPS ---")

        print(steps)

        subquestions_conclusions: list[...] = [
            self.conclude_step(
                step=s,
                query=query,
                n=n,
                filter_by_law=filter_by_law
            ) for s in steps
        ]

        self.THOUGHT_PROCESS["subquestions"] = subquestions_conclusions

        return subquestions_conclusions

    def main_query(
            self,
            query: str,
            n: int = 3,
            filter_by_law: list[str] | None = None,
    ):
        print(f"FILTER BY LAW: {filter_by_law}")

        CONCLUSIONS = self.subquestions_conclusion(
            query=query,
            n=n,
            filter_by_law=filter_by_law
        )

        CONCLUSIONS_TEMPLATE: FewShotPromptTemplate = \
            templates.conclusions_template(
                conclusions=CONCLUSIONS
            )

        final_answer = self.query_llm(
            prompt=CONCLUSIONS_TEMPLATE,
            query=query,
            tag="final_answer"
        )
        hashed_api_key = hashlib.sha256(self.openai_api_key.encode()).hexdigest()
        self.mongo_querier.insert_documents(self.prompts_sent, hashed_api_key)
        return {
            "final_answer": final_answer,
            "thought_process": self.THOUGHT_PROCESS
        }
