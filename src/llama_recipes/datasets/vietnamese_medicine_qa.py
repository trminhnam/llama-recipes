# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools


B_INST, E_INST = "[INST]", "[/INST]"


def tokenize_dialog(dialog, tokenizer):
    prompt_tokens = [
        tokenizer.encode(
            f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}",
            add_special_tokens=False,
        )
        for prompt in dialog[::2]
    ]
    answer_tokens = [
        tokenizer.encode(
            f"{answer['content'].strip()} {tokenizer.eos_token}",
            add_special_tokens=False,
        )
        for answer in dialog[1::2]
    ]
    dialog_tokens = list(
        itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens))
    )
    # Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [
        len(c)
        * [
            -100,
        ]
        if i % 2 == 0
        else c
        for i, c in enumerate(dialog_tokens)
    ]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"]))


def get_vietnamese_medicine_qa(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("tmnam20/VietnameseMedicineQA", split=split)


    dataset = dataset.map(
        lambda sample: {"dialog": [
            {
                "role": "user",
                "content": sample["question"],
            },
            {
                "role": "assistant",
                "content": sample["answer"],
            },
        ]},
        batched=False,
        remove_columns=list(dataset.features),
    )

    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset
