# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import wandb
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from veritas.training.utils import setup_sane_chat_format

# For single process logging: https://discuss.huggingface.co/t/limiting-print-and-log-statements/20035/2
logging.basicConfig(
    format="[%(asctime)s] - [%(levelname)s] - [%(name)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = get_logger(__name__, "INFO")


@dataclass
class SFTScriptArguments(ScriptArguments):
    # https://github.com/huggingface/trl/blob/47b9515fb1803cdfea3fd1e3a7167bec21c560e8/trl/models/utils.py#L78
    # https://huggingface.co/docs/trl/en/sft_trainer#add-special-tokens-for-chat-format
    # https://x.com/karpathy/status/1621578354024677377
    resize_to_multiple_of: int = field(
        default=64,
        metadata={"help": "Number to resize the embedding layer to. Defaults to None."},
    )
    # https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only
    mask_user_prompt: bool = field(
        default=False,
        metadata={"help": "Mask user prompt tokens using a DataCollatorForCompletionOnlyLM. Sets packing=False."},
    )
    chat_format: str = field(
        default="default",
        metadata={"help": "Which chat format(A combination of template & setup) to use: ['default', 'trl', 'sane']"},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run ID to use. If not specified, a new run will be created."},
    )


def batch_apply_chat_template(batch, tokenizer):
    # Convert "messages" -> "text" by applying chat_template
    batch["text"] = tokenizer.apply_chat_template(batch["messages"], add_generation_prompt=False, tokenize=False)
    return batch


def main(script_args, training_args, model_args):
    ################
    # WandB
    ################
    if script_args.wandb_run_id is not None:
        wandb.init(
            entity=os.environ["WANBD_ENTITY"],
            project=os.environ["WANDB_PROJECT"],
            id=script_args.wandb_run_id,
            resume="allow",
        )

    ################
    # Model
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        token=os.environ["HF_TOKEN"],
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()
    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        logger.info("Using AutoModelForImageTextToText")
        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        logger.info("Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    ###############
    # Tokenizer
    ###############
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
        token=os.environ["HF_TOKEN"],
    )

    # https://github.com/huggingface/transformers/issues/34842#issuecomment-2528550342
    # https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da?permalink_comment_id=4636728#gistcomment-4636728
    # https://github.com/huggingface/transformers/issues/26569#issuecomment-1768157610
    # https://huggingface.co/docs/transformers/llm_tutorial#padding-side
    # https://github.com/huggingface/transformers/issues/18388#issuecomment-1204369688
    tokenizer.padding_side = "right"

    ###############
    # Chat Template
    ###############
    # https://github.com/huggingface/trl/issues/1412
    if script_args.chat_format == "default" and tokenizer.chat_template is not None:
        # Default everything
        # Drop user provided pad token
        training_args.pad_token = None
        logger.warning("Using tokenizer's chat template. Make sure this is what you want.")
    elif script_args.chat_format == "default" and tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not have a chat template. Use 'trl' or 'sane'.")
    elif script_args.chat_format == "trl":
        if tokenizer.chat_template is not None:
            logger.warning("Tokenizer already has a chat template. Overriding it. Make sure this is what you want.")
            tokenizer.chat_template = None
        logger.info("Using TRL chat format to set CHATML chat template")
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer, format="chatml")
    elif script_args.chat_format == "sane":
        if tokenizer.chat_template is not None:
            logger.warning("Tokenizer already has a chat template. Overriding it. Make sure this is what you want.")
            tokenizer.chat_template = None
        logger.info(f"Using SANE chat format to set CHATML chat template with pad token: {training_args.pad_token}")
        model, tokenizer = setup_sane_chat_format(
            model=model,
            tokenizer=tokenizer,
            format="sane-chatml",
            token_mapping={"pad_token": training_args.pad_token},
        )
    else:
        raise NotImplementedError()

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ###############
    # Data Collator
    ###############
    # https://github.com/huggingface/trl/issues/632#issuecomment-1804440638
    # https://github.com/huggingface/trl/issues/588#issuecomment-2060688269
    # https://huggingface.co/docs/trl/v0.16.1/sft_trainer#using-tokenids-directly-for-responsetemplate
    if script_args.mask_user_prompt and script_args.chat_format == "default":
        raise NotImplementedError("Haven't implemented completions_only with default chat_format yet.")
    elif script_args.mask_user_prompt and script_args.chat_format != "default":
        # Do this after applying chat format
        # NOTE: Hardcoded key
        response_template_with_context = "<|im_end|>\n<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
        # Cannot using packing with this collator
        model_kwargs.pop("packing", None)
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids[2:], tokenizer=tokenizer)
        # Convert "messages" -> "text" because it's all manual now
        dataset = dataset.map(
            batch_apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            desc="Applying chat template",
        )
        # Don't need the messages column anymore; remove to avoid problems
        dataset = dataset.remove_columns(["messages"])
        logger.info("Using DataCollatorForCompletionOnlyLM with mask_user_prompt=True")
    elif not script_args.mask_user_prompt and script_args.chat_format != "default":
        logger.warning("Using default data collator when chat_format is not default")
    else:
        # Use default DataCollatorForLanguageModeling
        data_collator = None
        logger.warning("Using default data collator: DataCollatorForLanguageModeling. Make sure this is what you want.")

    ################
    # Sanity Checks
    ################
    if tokenizer.pad_token is None:
        raise ValueError("Please specify a PAD token. Your tokenizer still doesn't have one.")

    if tokenizer.pad_token == tokenizer.eos_token:
        logger.warning(f"Using PAD token == EOS token: {tokenizer.pad_token}. Make sure this is what you want")

    # Right for Tune, Left for Generate
    assert tokenizer.padding_side == "right"

    if script_args.mask_user_prompt and script_args.chat_format != "default":
        x_ct = dataset[script_args.dataset_train_split][9000:9002]["text"]
        logger.info("Check chat template:\n{}\n{}".format("\n".join(x_ct), "-" * 50))

        x_to = tokenizer(x_ct, padding="longest", truncation=False, return_tensors="pt")
        logger.info("Check tokenized values:\n{}\n{}".format(x_to["input_ids"], "-" * 50))

        x_de = [tokenizer.decode(t) for t in x_to["input_ids"]]
        logger.info("Check decoded version: \n{}\n{}".format("\n".join(x_de), "-" * 50))

        x_dc = data_collator(x_to["input_ids"])
        logger.info("Check data collator's working: \n{}\n{}".format(x_dc, "-" * 50))
    else:
        x = dataset[script_args.dataset_train_split][:2]["messages"]
        x_ct = tokenizer.apply_chat_template(x, tokenize=False)
        logger.info("Check application of chat template:\n{}\n{}".format("\n".join(x_ct), "-" * 50))

        x_to = tokenizer(x_ct, padding="longest", truncation=False, return_tensors="pt")
        logger.info("Check tokenized values:\n{}\n{}".format(x_to["input_ids"], "-" * 50))

        x_de = [tokenizer.decode(t) for t in x_to["input_ids"]]
        logger.info("Check decoded version: \n{}\n{}".format("\n".join(x_de), "-" * 50))

        # Convert dict of lists to list of dicts
        x_to_dc = [dict(zip(x_to.keys(), t)) for t in zip(*x_to.values())]
        x_dc = DataCollatorForLanguageModeling(tokenizer.pad_token_id)(x_to_dc)
        logger.info("Check data collator's working: \n{}\n{}".format(x_dc, "-" * 50))

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=data_collator,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (SFTScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
