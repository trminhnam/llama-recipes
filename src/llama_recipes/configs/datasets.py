# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = (
        "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    )


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class vietnamese_medicine_qa:
    dataset: str = "vietnamese_medicine_qa"
    train_split: str = "train"
    test_split: str = "validation"
    train_files: Tuple[str] = (
        "/media/ddien/minhnam/Vietnamese-medical-LLM/datasets/raw/duoc-lieu_train.json",
        "/media/ddien/minhnam/Vietnamese-medical-LLM/datasets/raw/thuoc_train.json",
    )
    validation_files: Tuple[str] = (
        "/media/ddien/minhnam/Vietnamese-medical-LLM/datasets/raw/duoc-lieu_val.json",
        "/media/ddien/minhnam/Vietnamese-medical-LLM/datasets/raw/thuoc_val.json",
    )
    