import ast
import asyncio
import logging
import re
import textwrap
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Dict, List, Optional, Tuple

from jinja2 import Template
from openai import AsyncOpenAI
from pydantic import Field, StringConstraints, create_model

logger = logging.getLogger("veritas")


@dataclass
class QGenConfig:
    structured: bool = False
    n_questions: int = 5
    n_queries: int = 5
    question_template: Optional[str] = None
    query_template: Optional[str] = None
    question_len_constraints: Optional[tuple[int, int]] = None
    query_len_constraints: Optional[tuple[int, int]] = None
    max_tokens: Optional[int] = 1024


@cache
def get_response_models(
    n_questions: int = 10,
    n_queries: int = 10,
    question_len_constraints: Optional[Tuple[int, int]] = None,
    query_len_constraints: Optional[Tuple[int, int]] = None,
):
    if question_len_constraints is None:
        QuestionResponseModel = create_model(
            "QuestionResponseModel",
            questions=(List[str], Field(..., min_length=n_questions, max_length=n_questions)),
        )
    else:
        QuestionString = Annotated[
            str,
            StringConstraints(min_length=question_len_constraints[0], max_length=question_len_constraints[1]),
        ]
        QuestionResponseModel = create_model(
            "QuestionResponseModel",
            questions=(List[QuestionString], Field(..., min_length=n_questions, max_length=n_questions)),
        )
    if query_len_constraints is None:
        QueryResponseModel = create_model(
            "QueryResponseModel",
            queries=(List[str], Field(..., min_length=n_queries, max_length=n_queries)),
        )
    else:
        QueryString = Annotated[
            str, StringConstraints(min_length=query_len_constraints[0], max_length=query_len_constraints[1])
        ]

        QueryResponseModel = create_model(
            "QueryResponseModel",
            queries=(List[QueryString], Field(..., min_length=n_queries, max_length=n_queries)),
        )

    return QuestionResponseModel, QueryResponseModel


def build_prompt_for_question_generation(claim: str, n_questions: int = 10, add_response_format: bool = False) -> str:
    template = Template(
        textwrap.dedent("""\
        You are an advanced fact-checking AI tasked with generating highly targeted, self-contained questions to help verify the truth of a specific claim.
        Your job is to generate **distinct**, **atomic**, and **fully self-contained** questions that, when answered, would provide **evidence** or **insight** into the claim's truth or falsehood.
        These questions will be used to retrieve relevant documents from a large text corpus, potentially in parallel, so each must stand completely on its own.

        ### Instructions
        1. Generate exactly {{ n_questions }} questions.
        2. Each question must:
        - Be **independent** and **non-redundant**.
        - Include **all necessary context from the original claim**—explicitly state relevant entities, numbers, dates, locations, etc. mentioned in the claim. **Do NOT use pronouns** (like "he", "she", "it", "they", "these", "those") or vague references that assume the reader knows the original claim.
        - Probe a **unique and essential** aspect of the claim's veracity.
        - Be **concise** but **complete**—include named entities and framing language from the claim.
        3. You are **not** fact-checking the claim or speculating—only generating investigative questions.

        ### Evaluation Checklist (for each question)
        1. Does it target a **specific verification angle** (e.g., source authenticity, context, supporting evidence)?
        2. Would answering it **help determine the truth** of the claim?
        3. Is it **standalone** and **clear**? Can someone understand exactly what is being asked without seeing the original claim? Does it avoid pronouns and ambiguous references?
        4. Is it **distinct** from the other questions?

        ### Example of Bad vs. Good Question
        Claim: "CDC data shows 290,956 excess deaths occurred between Jan. 26 and Oct. 3, 2020, with two-thirds directly attributed to COVID-19."
        BAD Question: "How many of *these* excess deaths were directly attributed to the coronavirus during the specified period?" (Uses "these", assumes context)
        GOOD Question: "According to CDC data, how many of the 290,956 excess deaths reported between Jan. 26 and Oct. 3, 2020, were directly attributed to COVID-19?" (Includes specific numbers and dates from the claim, avoids pronouns)

        ### Task
        Claim: "{{ claim }}"

        Generated Questions:""")
    )
    if add_response_format:
        # Note the newline before & after
        response_format = textwrap.dedent(f"""
        ### Response Format
        Output the final list of questions strictly in the format [[{", ".join(['"<question>"'] * n_questions)}]], where <question> is replaced by a single question.
        """)
    else:
        response_format = ""
    return template.render(n_questions=n_questions, claim=claim, response_format=response_format)


def build_prompt_for_query_generation(question: str, n_queries: int = 10, add_response_format: bool = False) -> str:
    template = Template(
        textwrap.dedent("""\
        You are a retrieval-optimization AI that transforms fact-checking questions into **concise search queries** for evidence gathering.

        ### Objective
        Your task is to generate **{{ n_queries }} distinct, highly concise, keyword-focused search queries** to retrieve documents that could help answer the investigative question below. The queries should be suitable for standard search engines.

        ### Key Goals
        - **Conciseness:** Queries should be short and direct, focusing on essential keywords.
        - **Keyword Focus:** Prioritize core entities, actions, and specific terms from the question.
        - **Diversity:** Use varied phrasings and related terms, but keep them brief.
        - **High Recall:** Aim to retrieve relevant documents effectively.

        ### Instructions
        For each query:
        - Make it **standalone** and **extremely concise** (typically 3-7 words).
        - Focus on **essential keywords** (nouns, verbs, proper nouns, key numbers/dates) likely to appear in relevant documents.
        - **Avoid full sentences or natural language questions.** Think in terms of search engine input.
        - Use strategies such as:
            - Extracting core entities and relationships (e.g., "Trump Billie Eilish destroying country documents")
            - Using specific names, locations, or numbers mentioned.
            - Trying synonyms or related concepts for key terms.
            - Including terms indicating evidence or claims (e.g., "report", "statement", "data", "evidence").
        - Ensure queries are **distinct** from each other.

        ### Example of Bad vs. Good Query
        Question: "According to CDC data, how many of the 290,956 excess deaths reported between Jan. 26 and Oct. 3, 2020, were directly attributed to COVID-19?"
        BAD Query: "What percentage of the excess deaths in the US between January and October 2020 were caused by COVID-19 according to the CDC?" (Too long, conversational)
        GOOD Query: "CDC excess deaths Jan-Oct 2020 COVID-19 attribution" (Concise, keyword-focused)
        GOOD Query: "290956 excess deaths COVID-19 cause CDC data" (Specific numbers, keywords)
        GOOD Query: "CDC report excess deaths 2020 COVID percentage" (Keywords, evidence term)

        ### Task
        Question: "{{ question }}"

        Generated Search Queries:""")
    )
    if add_response_format:
        # Note the newline before & after
        response_format = textwrap.dedent(f"""
        ### Response Format
        Output the final list of queries strictly in the format [[{", ".join(['"<query>"'] * n_queries)}]], where <query> is replaced by a single query.
        """)
    else:
        response_format = ""
    return template.render(n_queries=n_queries, question=question, response_format=response_format)


def parse_list_from_response(response_text: str) -> List[str]:
    pattern = re.compile(r"\[\[(.*?)\]\]")
    try:
        match = pattern.search(response_text, re.DOTALL)
        if not match:
            return []
        logger.debug(f"[QGEN][PARSE][MATCH]\n{match.group(1)}")

        # Attempt to safely evaluate the extracted string
        parsed = ast.literal_eval(f"[{match.group(1)}]")
        if not isinstance(parsed, list):
            raise ValueError("[QGEN][PARSE][ERROR]\nParsed data is not a list")

        return [item.strip() for item in parsed if isinstance(item, str)]

    except Exception as e:
        logger.debug(
            f"[QGEN][PARSE][ERROR]\nError parsing response.\n[RESPONSE TEXT]\n{response_text}\n[TRACEBACK]\n{repr(e)}"
        )
        return []


async def generate_questions_and_queries(
    claim: str,
    client: AsyncOpenAI,
    model_id: str,
    max_retries: int,
    cfg: QGenConfig,
) -> Dict[str, List[str]]:
    logger.debug(f"Generating questions and queries for claim '{claim}'")

    # Get response models
    if cfg.structured:
        logger.debug("Using structured response models")
        QuestionResponseModel, QueryResponseModel = get_response_models(
            cfg.n_questions,
            cfg.n_queries,
            question_len_constraints=cfg.question_len_constraints,
            query_len_constraints=cfg.query_len_constraints,
        )
    else:
        QuestionResponseModel, QueryResponseModel = None, None

    # Get prompt for question generation
    if cfg.question_template:
        q_prompt = Template(cfg.question_template).render(claim=claim, n_questions=cfg.n_questions)
    else:
        q_prompt = build_prompt_for_question_generation(claim, cfg.n_questions, add_response_format=not cfg.structured)
    logger.debug(f"[QGEN][PROMPT]\n{q_prompt}")

    # Generate questions
    question_response = await client.chat.completions.create(  # type: ignore
        model=model_id,
        response_model=QuestionResponseModel,
        messages=[{"role": "user", "content": q_prompt}],
        max_retries=max_retries,
        max_tokens=cfg.max_tokens,
    )
    # Parse questions from response
    if cfg.structured:
        logger.debug(f"[QGEN][RESPONSE]\n{question_response}")
        questions = question_response.questions
    else:
        r = question_response.choices[0].message.content
        logger.debug(f"[QGEN][RESPONSE]\n{r}")
        questions = parse_list_from_response(r)

    # Prepare tasks for concurrent query generation
    query_gen_tasks = []
    for question in questions:
        # Get prompt for query generation
        if cfg.query_template:
            prompt = Template(cfg.query_template).render(question=question, n_queries=cfg.n_queries)
        else:
            prompt = build_prompt_for_query_generation(question, cfg.n_queries, add_response_format=not cfg.structured)
        logger.debug(f"[QGEN][PROMPT]\n{prompt}")

        # Create a task for generating queries for this question
        task = client.chat.completions.create(  # type: ignore
            model=model_id,
            response_model=QueryResponseModel,
            messages=[{"role": "user", "content": prompt}],
            max_retries=max_retries,
            max_tokens=cfg.max_tokens,
        )
        query_gen_tasks.append(task)

    # Execute query generation tasks concurrently
    query_responses = await asyncio.gather(*query_gen_tasks)

    # Parse queries from responses
    all_queries = []
    for response in query_responses:
        if cfg.structured:
            logger.debug(f"[QGEN][RESPONSE]\n{response}")
            queries = response.queries
        else:
            r = response.choices[0].message.content
            logger.debug(f"[QGEN][RESPONSE]\n{r}")
            queries = parse_list_from_response(r)
        all_queries.append(queries)

    return {question: queries for question, queries in zip(questions, all_queries)}
