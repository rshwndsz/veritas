import logging
import re
import textwrap
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Annotated, Dict, List, Optional

from jinja2 import Template
from openai import AsyncOpenAI
from pydantic import Field, StringConstraints, create_model

logger = logging.getLogger("veritas")


@dataclass
class JudgeConfig:
    structured: bool = False
    template: Optional[str] = None
    reasoning_len_constraints: Optional[tuple[int, int]] = None
    max_tokens: Optional[int] = 2048


class VerdictLabel(Enum):
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    CONFLICTING = "Conflicting"
    NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"


@cache
def get_response_model(reasoning_len_constraints: Optional[tuple[int, int]] = None):
    if reasoning_len_constraints is None:
        ReasoningString = Annotated[str]
    else:
        ReasoningString = Annotated[
            str,
            StringConstraints(min_length=reasoning_len_constraints[0], max_length=reasoning_len_constraints[1]),
        ]
    VerdictResponseModel = create_model(
        "VerdictResponseModel",
        reasoning=(ReasoningString, Field(..., min_length=0, max_length=512)),
        label=(VerdictLabel, Field(...)),
    )
    return VerdictResponseModel


def build_prompt_for_verdict(claim: str, qa_pairs: List[Dict[str, str]], add_response_format: bool = True) -> str:
    template = Template(
        textwrap.dedent("""\
        You are an advanced fact-checking AI. Your task is to determine the veracity of a given CLAIM based *strictly* on the provided EVIDENCE (a series of Question-Answer pairs).

        ### Inputs
        CLAIM: "{{ claim }}"

        EVIDENCE:
        {% for qa in qa_pairs %}
        --- Evidence Piece {{ loop.index }} ---
        QUESTION: {{ qa.question }}
        ANSWER: {{ qa.answer }}
        {% endfor %}
        --- End of Evidence ---

        ### Task Steps
        1.  **Analyze Evidence:** Carefully review each Question-Answer pair provided in the EVIDENCE section.
        2.  **Synthesize Findings:** Combine the information from ALL evidence pieces. Explicitly state how the collective evidence supports or contradicts the *entire* CLAIM. Identify which specific parts of the claim are addressed by the evidence and whether they are confirmed, refuted, or if the evidence is insufficient/conflicting regarding those parts.
        3.  **Determine Final Verdict:** Based *only* on your synthesis in Step 2, assign ONE of the labels below. Your reasoning must clearly justify the chosen label by referencing the evidence synthesis.

        ### Labels & Definitions
        - "Supported": ALL essential elements of the claim are clearly and unambiguously supported by the combined evidence.
        - "Refuted": At least one essential element of the claim is clearly and unambiguously contradicted by the combined evidence.
        - "Conflicting": Use this **only if two or more evidence pieces (Answers) provide contradictory information about the *same* essential element of the claim.** Do not use this if evidence simply contradicts the claim itself but not other evidence pieces.
        - "Not Enough Evidence": Use this **only if the combined evidence is insufficient, irrelevant, or too ambiguous to clearly determine if the essential elements of the claim are supported or refuted.**

        ### Labeling Priority & Strictness
        - First, determine if the evidence clearly points to **Supported** or **Refuted**.
        - If not, check for **Conflicting** evidence as defined above.
        - If none of the above apply, use **Not Enough Evidence**.
        - **Crucially:** Your final verdict label MUST be a direct logical consequence of your synthesis (Step 2) and based *solely* on the provided EVIDENCE. Do not introduce external information or assumptions.

        {{ response_format }}

        ### Instructions
        - Follow the Task Steps meticulously.
        - Ensure your synthesis explicitly links the evidence pieces to the claim's components.
        - Your final verdict must be one of the four defined labels and nothing else.

        ANSWER:""")
    )
    if add_response_format:
        response_format = textwrap.dedent("""\
        ### Response Format
        After "ANSWER:" do the following:
        - Think step by step and synthesize an answer based on the evidence.
        - While your reasoning can be complex, your final answer must be clear, as concise as possible, and directly address the question and without slop.
        - Strictly follow the following format: Enclose your thinking in <think></think> and then enclose your final answer in <answer></answer>.

        Here's an example template of the response:
        <think>
        [Your reasoning process here]
        </think>
        <answer>
        [Your final answer here]
        </answer>""")
    else:
        response_format = ""
    prompt = template.render(claim=claim, qa_pairs=qa_pairs, response_format=response_format)
    return prompt


def parse_verdict(response_text: str) -> Dict[str, Optional[str]]:
    reasoning, verdict = None, None

    pattern = re.compile(r"<think\b[^>]*>(.*?)</think>\s*<answer\b[^>]*>(.*?)</answer>")
    match = pattern.search(response_text, re.IGNORECASE | re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        verdict = match.group(2).strip()

    for label in [v.value for v in VerdictLabel]:
        if re.search(rf"\b{re.escape(label)}\b", response_text, re.IGNORECASE):
            verdict = label
            logger.warning(f"Could not parse <think>/<answer> tags; falling back to label match: {verdict}")
            break

    if verdict is None:
        logger.error("[JUDG][PARSE][ERROR] Failed to extract verdict label or reasoning.")

    return {"reasoning": reasoning, "verdict": verdict}


async def generate_verdict(
    claim: str,
    qa_pairs: List[Dict[str, str]],
    client: AsyncOpenAI,
    model_id: str,
    max_retries: int,
    cfg: JudgeConfig,
) -> Optional[str]:
    # Build prompt
    if cfg.template:
        prompt = Template(cfg.template).render(claim=claim, qa_pairs=qa_pairs)
    else:
        prompt = build_prompt_for_verdict(claim=claim, qa_pairs=qa_pairs, add_response_format=not cfg.structured)
    logger.debug(f"[JUDG][PROMPT]\n{prompt}")

    # Get response model
    if cfg.structured:
        response_model = get_response_model(cfg.reasoning_len_constraints)
    else:
        response_model = None

    # Generate verdict
    response = await client.chat.completions.create(  # type: ignore
        model=model_id,
        response_model=response_model,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries,
        max_tokens=cfg.max_tokens,
    )

    # Parse response
    if cfg.structured:
        logger.debug(f"[JUDG][RESPONSE]\n{response}")
        verdict = response.label
    else:
        r = response.choices[0].message.content
        logger.debug(f"[JUDG][RESPONSE]\n{r}")
        parsed = parse_verdict(r)
        verdict = parsed.get("verdict", None)
        if verdict is not None:
            verdict = VerdictLabel(verdict)

    # Convert to string from enum
    if verdict is not None:
        verdict = verdict.value

    return verdict
