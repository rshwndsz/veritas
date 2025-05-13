import logging
import re
import textwrap
from dataclasses import dataclass
from functools import cache
from typing import Annotated, List, Optional

from jinja2 import Template
from openai import AsyncOpenAI
from pydantic import Field, StringConstraints, create_model

from veritas.kb.chunk import Chunk

logger = logging.getLogger("veritas")


@dataclass
class AGenConfig:
    structured: bool = False
    template: Optional[str] = None
    reasoning_len_constraints: Optional[tuple[int, int]] = None
    answer_len_constraints: Optional[tuple[int, int]] = None
    max_tokens: Optional[int] = 2048


@dataclass
class EvidenceItem:
    question: str
    answer: str | None
    url: str
    scraped_text: str


@dataclass
class ParsedResponse:
    reasoning: Optional[str]
    answer: Optional[str]
    best_chunk_number: int


@cache
def get_response_model(
    reasoning_len_constraints: Optional[tuple[int, int]] = None,
    answer_len_constraints: Optional[tuple[int, int]] = None,
):
    if reasoning_len_constraints is None:
        ReasoningString = Annotated[str]
    else:
        ReasoningString = Annotated[
            str,
            StringConstraints(min_length=reasoning_len_constraints[0], max_length=reasoning_len_constraints[1]),
        ]

    if answer_len_constraints is None:
        AnswerString = Annotated[str]
    else:
        AnswerString = Annotated[
            str,
            StringConstraints(min_length=answer_len_constraints[0], max_length=answer_len_constraints[1]),
        ]
    AnswerResponseModel = create_model(
        "AnswerResponseModel",
        reasoning=(ReasoningString, Field(..., min_length=0, max_length=512)),
        answer=(AnswerString, Field(..., min_length=0, max_length=256)),
        best_chunk_number=(int, Field(..., ge=1)),
    )
    return AnswerResponseModel


def build_prompt_for_answer_synthesis(question: str, chunks: List[Chunk], add_response_format: bool = True) -> str:
    template = Template(
        textwrap.dedent("""\
    You are an advanced fact-checking AI, tasked with answering questions based *strictly* on provided evidence chunks.
    You have been given a question and a set of evidence chunks retrieved from various sources.
    Your goal is to **synthesize a single, comprehensive, and well-reasoned answer** by combining relevant information from *all* applicable evidence chunks.
    You must ground your answer *only* in the evidence provided and avoid speculation or unsupported claims.
    Name your sources journalistically.

    ### Instructions
    1.  **Analyze All Evidence:** Read and understand every provided evidence chunk.
    2.  **Identify Relevant Information:** Pinpoint information across *all* chunks that directly addresses the question. Note agreements, contradictions, or complementary details between chunks.
    3.  **Synthesize a Coherent Answer:**
        *   Combine the relevant pieces of information from the necessary chunks into a single, unified answer.
        *   If multiple chunks contribute, integrate their information logically. Do *not* just pick one chunk's answer if others provide relevant context or details.
        *   If chunks conflict, acknowledge the discrepancy if necessary to answer accurately, citing the differing sources.
    4.  **Cite Sources:** Distill a short source name from each URL (e.g., "New York Times", "BBC News"). When presenting information in your answer, cite the corresponding source name(s) (e.g., "According to the New York Times...", "BBC News reported..."). Do *not* use the word "chunk" or the raw URL in the final answer.
    5.  **Evaluate Credibility:** Use your world knowledge to assess the credibility of sources, giving more weight to reputable outlets, especially when synthesizing potentially conflicting information. State this weighting in your reasoning if relevant.
    6.  **Select Best Chunk:** After formulating your synthesized answer, identify the index (starting from 1) of the single chunk that provides the *most critical piece of evidence* supporting your final answer.
    7.  **Be Concise and Factual:** Ensure your final answer is direct, avoids unnecessary jargon, and strictly adheres to the information within the provided chunks. Avoid speculation or outside information.

    {{response_format}}
    ### Evidence Chunks:
    {% for chunk in chunks %}
    [CHUNK [{{ loop.index }}] START]
    URL: {{ chunk.source_url }}
    CONTENT: {{ chunk.content }}
    [CHUNK [{{ loop.index }}] END]
    {% endfor %}

    ### Task
    QUESTION: "{{ question }}"

    ANSWER:""")
    )
    if add_response_format:
        # Note the newlines before and after
        response_format = textwrap.dedent("""
        ### Response Format
        First, think step by step within <think>...</think>. This is your reasoning process.
        Then, clearly state your final synthesized answer within <answer>...</answer>.
        Then, provide the index of the chunk that best supports your answer after CHUNK=

        Here's an example template of the response:
        <think>
        [Your reasoning process here]
        </think>
        <answer>
        [Your final answer here]
        </answer>
        CHUNK=[index of the best chunk that supports your answer here]
        """)
    else:
        response_format = ""
    prompt = template.render(question=question, chunks=chunks, response_format=response_format)
    return prompt


def parse(response_text: str) -> ParsedResponse:
    # Parse <think>{reasoning}</think> <answer>{answer}</answer> CHUNK={index}
    # Return None if not found
    match = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*CHUNK=(\d+)", response_text, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
        try:
            best_chunk_number = max(int(match.group(3).strip()), 1)
        except ValueError:
            logger.warning(
                "[AGEN][PARSE][ERROR]\nFailed to parse chunk index from response. Using the first chunk as best chunk",
            )
            best_chunk_number = 1
        return ParsedResponse(
            reasoning=reasoning,
            answer=answer,
            best_chunk_number=best_chunk_number,
        )
    else:
        logger.warning(
            "[AGEN][PARSE][ERROR]\nFailed to parse response for reasoning and response. "
            "Using the entire response as answer",
        )
        return ParsedResponse(
            reasoning=None,
            answer=response_text,
            best_chunk_number=1,
        )


async def synthesize_answer(
    question: str,
    chunks: List[Chunk],
    client: AsyncOpenAI,
    model_id: str,
    max_retries: int,
    cfg: AGenConfig,
) -> EvidenceItem:
    if not chunks:
        logger.warning("[AGEN][ERROR] No chunks provided for answer synthesis. Returning empty result.")
        return EvidenceItem(
            question=question,
            answer=None,
            url="",
            scraped_text="",
        )

    # Build prompt
    if cfg.template:
        prompt = Template(cfg.template).render(question=question, chunks=chunks)
    else:
        prompt = build_prompt_for_answer_synthesis(
            question=question, chunks=chunks, add_response_format=not cfg.structured
        )
    logger.debug(f"[AGEN][PROMPT]\n{prompt}")

    # Get response model
    if cfg.structured:
        response_model = get_response_model(cfg.reasoning_len_constraints, cfg.answer_len_constraints)
    else:
        response_model = None

    # Generate answer & reasoning
    response = await client.chat.completions.create(  # type: ignore
        model=model_id,
        response_model=response_model,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries,
        max_tokens=cfg.max_tokens,
    )

    # Parse response
    if cfg.structured:
        logger.debug(f"[AGEN][RESPONSE]\n{response}")
        # Convert from 1-indexed to 0-indexed if valid
        if response.best_chunk_number > len(chunks) or response.best_chunk_number < 1:
            logger.warning(
                "[AGEN][PARSE][ERROR]\nParsed chunk index is out of range. Using the first chunk as best chunk",
            )
            best_chunk_index = 0
        else:
            best_chunk_index = max(0, response.best_chunk_number - 1)
        result = EvidenceItem(
            question=question,
            answer=response.answer,
            url=chunks[best_chunk_index].source_url,
            scraped_text=chunks[best_chunk_index].content,
        )
    else:
        response_text = response.choices[0].message.content
        logger.debug(f"[AGEN][RESPONSE]\n{response_text}")
        parsed: ParsedResponse = parse(response_text)
        if parsed.best_chunk_number > len(chunks) or parsed.best_chunk_number < 1:
            logger.warning(
                "[AGEN][PARSE][ERROR]\nParsed chunk index is out of range. Using the first chunk as best chunk",
            )
            best_chunk_index = 0
        else:
            best_chunk_index = max(0, parsed.best_chunk_number - 1)
        result = EvidenceItem(
            question=question,
            answer=parsed.answer,
            url=chunks[best_chunk_index].source_url,
            scraped_text=chunks[best_chunk_index].content,
        )

    return result
