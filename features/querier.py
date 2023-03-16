#!/usr/bin/env python3

from typing import TypedDict
import re
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
from features.chain import Chain
import os

from features.similarity_queries import similarity_query
from features import chatgpt

from langchain.chat_models import ChatOpenAI

from features import templates


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

    def instance_external_resources(self):
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key
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
        **kwargs,
    ) -> list[str] | str:
        """
        if pltags is not None:
            self.llm.pl_tags = pltags
        """

        chain = LLMChain(llm=self.llm, prompt=prompt)
        results = initial + chain.run(**kwargs)

        if initial and prefix:
            return self._parse(
                results=results.splitlines(),
                prefix=prefix
            )
        else:
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
        print("#" * 50)
        print("> UNIQ SECTIONS MATCHED:")
        print(uniq_sections_matched)
        print("#" * 50)

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
            return self.summarize_big_similar_sections(
                docs=docs,
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

        summarize_template: PromptTemplate =\
            templates.summarize_similar_sections_template()

        return self.query_llm(
            prompt=summarize_template,
            query=query,
            sections=bulleted_similar_sections,
            pltags=["summarize similar_sections (fast)"]
        )

    def summarize_big_similar_sections(
        self,
        docs,
        query
            ) -> str:
        return ""

    """
    DISABLED AND MARKED FOR REFACTOR.
    def summarize_big_similar_sections(self, docs, query):
        question_prompt, refine_prompt =\
            templates.summarize_big_similar_sections_templates()
        self.llm.pl_tags = ["summarize similar sections (slow)"]
        chain = load_summarize_chain(
            self.llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
        )

        return chain.run(input_documents=docs, query=query)
    """

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
            pltags=["subquestions queries"]
        )

        answers: list[templates.Answer] = self._pmap(
            self.answer_question,
            questions_list,
            n,
            filter_by_law,
        )

        ANSWERS_TEMPLATE: FewShotPromptTemplate =\
            templates.answers_template(answers=answers)

        conclusion = self.query_llm(
            prompt=ANSWERS_TEMPLATE,
            query=query,
            step=step,
            pltags=["answer subquestion"]
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
            pltags=["get subquestions"]
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

        CONCLUSIONS_TEMPLATE: FewShotPromptTemplate =\
            templates.conclusions_template(
                conclusions=CONCLUSIONS
            )

        final_answer = self.query_llm(
            prompt=CONCLUSIONS_TEMPLATE,
            query=query,
            pltags=["final answer"]
        )

        return {
            "final_answer": final_answer,
            "thought_process": self.THOUGHT_PROCESS
        }
