import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from accelerate.logging import get_logger
from transformers import PreTrainedModel, PreTrainedTokenizer

logging.basicConfig(
    format="%(asctime)s - %(name)24s - %(levelname)7s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = get_logger(__name__, "INFO")


@dataclass
class SaneChatMlSpecialTokens:
    """
    Dataclass for ChatML special tokens.

    Attributes:
        bos_token (str): Token marking the beginning of the conversation.
                         Defaults to <|im_start|> unless overridden.
        eos_token (str): Token marking the end of the conversation.
                         Defaults to <|im_end|> unless overridden.
        pad_token (str): Padding token. Falls back to eos_token if not provided.
        im_start_token (str): Turn-level start token (always <|im_start|>).
        im_end_token (str): Turn-level end token (always <|im_end|>).
    """

    bos_token: str
    eos_token: str
    pad_token: str
    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"

    @property
    def system(self) -> str:
        return f"{self.im_start_token}system"

    @property
    def user(self) -> str:
        return f"{self.im_start_token}user"

    @property
    def assistant(self) -> str:
        return f"{self.im_start_token}assistant"

    @property
    def chat_template(self) -> str:
        """
        Returns a Jinja template string for chat formatting.
        - If the conversation-level bos_token differs from the turn token, it is printed at the beginning
          when the first message's role is 'user' or 'system'.
        - Each message is wrapped in turn tokens.
        - At the end, if add_generation_prompt is true, a new turn for 'assistant' is started.
          Else if the last message is by 'assistant', the conversation-level eos_token is appended
          (only if eos_token differs from the turn token).
        """
        template = ""
        # Beginning: show conversation bos_token if defined differently from im_start_token.
        if self.bos_token != self.im_start_token:
            template += (
                "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}"
                "{{ '" + self.bos_token + "' }}"
                "{% endif %}"
            )
        # For each message, wrap with turn tokens.
        template += (
            "{% for message in messages %}"
            "{{ '"
            + self.im_start_token
            + "' + message['role'] + '\\n' + message['content'] + '"
            + self.im_end_token
            + "\\n' }}"
            "{% endfor %}"
        )
        # End: depending on context, add a generation prompt or finish with conversation eos_token
        if self.eos_token != self.im_end_token:
            template += (
                "{% if add_generation_prompt %}"
                "{{ '" + self.im_start_token + "assistant\\n' }}"
                "{% elif messages[-1]['role'] == 'assistant' %}"
                "{{ '" + self.eos_token + "' }}"
                "{% endif %}"
            )
        else:
            template += "{% if add_generation_prompt %}{{ '" + self.im_start_token + "assistant\\n' }}{% endif %}"
        return template


def get_sane_chatml_special_tokens(
    tokenizer: PreTrainedTokenizer,
    token_mapping: Optional[Dict[str, str]] = None,
) -> SaneChatMlSpecialTokens:
    """
    Create a SaneChatMlSpecialTokens instance.

    For the conversation-level tokens:
      - Defaults are <|im_start|> for bos_token and <|im_end|> for eos_token.
      - If the tokenizer already defines them (trained tokens), or if a token_mapping is provided,
        those values are used.

    The pad_token defaults to the eos_token unless otherwise defined.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to check for preexisting tokens.
        token_mapping (Optional[Dict[str, str]]): Optional mapping with keys 'bos_token', 'eos_token',
            and optionally 'pad_token'.

    Returns:
        SaneChatMlSpecialTokens: The constructed special tokens.
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    # Use values from token_mapping if available.
    conv_bos = token_mapping.get("bos_token") if token_mapping and "bos_token" in token_mapping else None
    conv_eos = token_mapping.get("eos_token") if token_mapping and "eos_token" in token_mapping else None
    conv_pad = token_mapping.get("pad_token") if token_mapping and "pad_token" in token_mapping else None

    # For conversation-level tokens, prefer: mapping > tokenizer > defaults.
    bos_token = conv_bos or tokenizer.bos_token or im_start
    eos_token = conv_eos or tokenizer.eos_token or im_end
    pad_token = conv_pad or tokenizer.pad_token or eos_token

    return SaneChatMlSpecialTokens(
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
    )


FORMAT_MAPPING = {"sane-chatml": get_sane_chatml_special_tokens}


def setup_sane_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml"]] = "chatml",
    token_mapping: Optional[Dict[str, str]] = None,
    resize_to_multiple_of: Optional[int] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Set up the ChatML format.

    Conversation-level tokens (bos_token, eos_token, pad_token) are determined by:
      - The provided token_mapping (if any),
      - Or the tokenizer's already defined tokens (if any),
      - Otherwise default to im_start/im_end for bos/eos (and pad falls back to eos_token).

    Turn-level tokens (im_start, im_end) are always added.

    The chat template is built accordingly.

    Args:
        model (PreTrainedModel): The model to modify.
        tokenizer (PreTrainedTokenizer): The tokenizer to modify.
        format (Optional[Literal["chatml"]]): The format to apply (only "chatml" is currently supported).
        token_mapping (Optional[Dict[str, str]]): Optional mapping to override conversation-level tokens.
        resize_to_multiple_of (Optional[int]): Optional resizing parameter for model embeddings.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]: The updated model and tokenizer.
    """
    if tokenizer.chat_template is not None:
        raise ValueError(
            "Chat template is already added to the tokenizer. To overwrite it, please set tokenizer.chat_template to None."
        )

    if format not in FORMAT_MAPPING:
        raise ValueError(f"Unsupported format: {format}. Supported: {list(FORMAT_MAPPING.keys())}")

    # Build our chat format tokens (using mapping if provided)
    chat_format = FORMAT_MAPPING[format](tokenizer, token_mapping)

    # Select the tokens to add
    # Add turn-level tokens unconditionally
    # If BOS & EOS exist, they are already added
    # If they don't, BOS = im_start, EOS = im_end
    # PAD has to be added if it is not already present & not equal to EOS
    # That's why we do this before updating the tokenizer's special tokens
    additional_tokens = [chat_format.im_start_token, chat_format.im_end_token]
    if chat_format.pad_token != chat_format.eos_token and chat_format.pad_token != tokenizer.pad_token:
        additional_tokens.append(chat_format.pad_token)

    # Always assign (they might be already set but now follow our mapping/default resolution)
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token

    tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})

    # Set the chat template
    tokenizer.chat_template = chat_format.chat_template

    # Resize model token embeddings.
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )

    logger.info(
        f"Resizing done. (BOS, EOS, PAD) = {tokenizer.bos_token_id}, {tokenizer.eos_token_id}, {tokenizer.pad_token_id}"
    )

    # Update model configuration.
    if getattr(model, "config", None) is not None:
        logger.info("Updating model config with tokenizer special tokens.")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    if getattr(model, "generation_config", None):
        logger.info("Updating model generation config with tokenizer special tokens.")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer
