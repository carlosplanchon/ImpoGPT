#!/usr/bin/env python3

from typing import cast, TypedDict

from langchain import PromptTemplate, FewShotPromptTemplate


class Answer(TypedDict):
    question: str
    answer: str


class Conclusion(TypedDict):
    step: str
    conclusion: str


def dedent(text: str) -> str:
    """A more lenient version of `textwrap.dedent`."""
    return "\n".join(map(str.strip, text.splitlines())).strip()


def queries_template() -> PromptTemplate:
    """The template to determine sub-question queries."""

    return PromptTemplate(
        template=dedent(
            """
            Su tarea es determinar un conjunto de consultas en lenguaje natural para responder una pregunta.
            Las preguntas corren contra una base de datos compilada sobre legislación.

            La pregunta general que usted está intentando responder es: {query}
            Usted está en el paso: {step}

            Genere una lista con viñetas con máximo {n} consultas en lenguaje natural para completar este paso.
            -
            """
        ),
        input_variables=["query", "step", "n"],
    )


def steps_template() -> PromptTemplate:
    """The template to determine sub-questions."""

    return PromptTemplate(
        template=dedent(
            """
            Usted es un chatbot legal. Su trabajo es ayudar a los abogados a navegar por cuestiones legales.
            Su tarea es determinar un conjunto de máximo {n} subpreguntas que responderían una pregunta.
            Usted no tiene que responder a la pregunta, simplemente determine la mejor manera de responderla.

            Por ejemplo, si la pregunta es "¿Cuál es el color de casa más popular?":
            1. Determine todos los colores posibles que puede tener una casa.
            2. Determine el número de casas que son de cada color.
            3. Determine el color más popular.

            La pregunta es: {query}

            1.
            """
        ),
        input_variables=["query", "n"],
    )


def answers_template(answers: list[Answer]) -> FewShotPromptTemplate:
    """The template to summarize sub-questions."""

    return FewShotPromptTemplate(
        examples=cast(list[dict], answers),
        example_prompt=PromptTemplate(
            template=dedent(
                """
                Pregunta:
                {question}

                Respuesta:
                {answer}
            """
            ),
            input_variables=["question", "answer"],
        ),
        prefix=dedent(
            """
            Tu tarea consiste en tomar una serie de preguntas y respuestas, y usarla para completar un paso para responder la pregunta original.

            La pregunta original que usted está tratando de responder es: {query}
            Usted está en el paso: {step}

            Aqui están las preguntas y respuestas:
            """
        ),
        suffix="Paso completo:\n",
        input_variables=["query", "step"],
    )


def conclusions_template(conclusions: list[Conclusion]):
    """The template to summarize the final answer from a set of conclusions."""

    return FewShotPromptTemplate(
        examples=cast(list[dict], conclusions),
        example_prompt=PromptTemplate(
            template=dedent(
                """
                Paso:
                {step}

                Conclusión:
                {conclusion}
            """
            ),
            input_variables=["step", "conclusion"],
        ),
        prefix=dedent(
            """
            Tu tarea es tomar una serie de pasos efectuados con el fin de responder una pregunta, y usarlos para responder esa pregunta.

            Responda la pregunta de forma estructurada, usando el formato especificado. Por ejemplo, si la pregunta especifica una lista de propiedades, dibuja una tabla con esa lista.

            La pregunta que usted está intentando responder: {query}

            Los pasos que usted ha seguido para responder a esta pregunta son:
            """
        ),
        suffix="Final answer:\n",
        input_variables=["query"],
    )


def summarize_similar_sections_template():
    return PromptTemplate(
        template=dedent(
            """
            Una serie de artículos legislativos fueron encontrados para intentar responder a la pregunta: "{query}".
            Tu trabajo es responder la pregunta resumiendo las respuestas de todas las secciones encontradas.

            Las secciones son:
            
            {sections}
            
            El resumen es:
            """
        ),
        input_variables=["query", "sections"],
    )


def summarize_big_similar_sections_templates():
    question_prompt = PromptTemplate(
        template=dedent(
            """
            Una serie de artículos legislativos fueron encontrados para intentar responder a la pregunta: "{query}".
            Tu trabajo es responder la pregunta resumiendo las respuestas de todas las secciones encontradas.

            El resumen hasta este momento es:
            <empty>

            Aquí está la siguiente sección:

            {text}

            El nuevo resumen hasta este momento es:
            """
        ),
        input_variables=["text", "query"],
    )
    refine_prompt = PromptTemplate(
        template=dedent(
            """
            Una serie de artículos legislativos fueron encontrados para intentar responder a la pregunta: "{query}".
            Tu trabajo es responder la pregunta resumiendo las respuestas de todas las secciones encontradas.

            El resumen hasta este momento es:
            {existing_answer}

            Aquí está la siguiente sección:

            {text}

            El nuevo resumen hasta este momento es:
            """
        ),
        input_variables=["text", "query", "existing_answer"],
    )
    return question_prompt, refine_prompt
