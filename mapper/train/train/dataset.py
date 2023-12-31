from datasets import load_dataset, IterableDatasetDict, concatenate_datasets, interleave_datasets
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import torch
from typing import Union, Optional
from dataclasses import dataclass
import random


def bernoulli(probability: float) -> bool:
    return random.random() < probability


@dataclass
class DataCollatorForEL(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    embedding_size: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, batch_features):
        # Tokenize just the input features (i.e., the input text).
        input_features = [{
            k: v
            for k, v in features.items()
            if k != "targets" and k != "spans" and k != "labels"
        } for features in batch_features]
        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        spans_max_length = max(
            len(features["spans"]) for features in batch_features)

        batch_spans = torch.stack([
            torch.cat((feature["spans"],
                       torch.zeros(spans_max_length - len(feature["spans"]),
                                   2,
                                   dtype=torch.int64)),
                      dim=0) for feature in batch_features
        ], 0)
        batch_targets = torch.stack([
            torch.cat((feature["targets"],
                       torch.zeros(spans_max_length - len(feature["spans"]),
                                   self.embedding_size,
                                   dtype=torch.int64)),
                      dim=0) for feature in batch_features
        ], 0)

        batch["spans"] = batch_spans
        batch["targets"] = batch_targets

        return batch


def prepare_features(examples, tokenizer, max_length, doc_stride, embeddings):
    tokenized_examples = tokenizer(
        examples["context"],
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["spans"] = []
    tokenized_examples["targets"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]

        spans = []
        targets = []

        for span in examples["anchors"][sample_index]:
            span_start, span_end, qid = span["start"], span["end"], span["qid"]

            if qid is None:
                continue

            embedding = embeddings[qid]

            if embedding is None:
                continue
            # if "tag" in span and span["tag"] == 4:
            # continue

            # Start token index of the current span in the text.
            token_start_index = 0

            while offsets[token_start_index][0] == 0 and offsets[
                    token_start_index][1] == 0:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1

            while offsets[token_end_index][0] == 0 and offsets[
                    token_end_index][1] == 0:
                token_end_index -= 1

            # Detect if the span is out of the sequence length.
            if (offsets[token_start_index][0] <= span_start
                    and offsets[token_end_index][1] >= span_end):
                # Move the token_start_index and token_end_index to the two ends of the span.
                # Note: we could go after the last offset if the span is the last word (edge case).
                try:
                    while offsets[token_start_index][0] < span_start:
                        token_start_index += 1

                    while offsets[token_end_index][1] > span_end:
                        token_end_index -= 1
                except Exception:
                    continue

                spans.append((token_start_index, token_end_index))
                targets.append(torch.tensor(embedding))

        if len(spans) > 0:
            spans, targets = zip(
                *sorted(zip(spans, targets), key=lambda x: x[0]))

        for x, y in spans:
            if bernoulli(0.8):
                input_ids[x:y + 1] = [
                    tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                ] * (y + 1 - x)
            elif bernoulli(0.5):
                input_ids[x:y + 1] = [
                    random.randint(0, tokenizer.vocab_size - 1)
                    for _ in range(y + 1 - x)
                ]

        tokenized_examples["spans"].append(list(spans))
        tokenized_examples["targets"].append(list(targets))

    return tokenized_examples


def get_dataset_wikianc(tokenizer, embeddings, languages):
    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train_shards = [
        load_dataset("cyanic-selkie/wikianc", language, split="train")
        for language in languages
    ]
    train = concatenate_datasets(train_shards).shuffle(seed=42)
    train_total = len(train)

    validation_shards = [
        load_dataset("cyanic-selkie/wikianc", language, split="validation")
        for language in languages
    ]
    validation = concatenate_datasets(validation_shards)
    validation_total = len(validation)

    dataset = IterableDatasetDict({
        "train":
        train.to_iterable_dataset(),
        "validation":
        validation.to_iterable_dataset()
    })

    dataset = dataset.remove_columns([
        "uuid", "article_title", "article_pageid", "article_qid",
        "section_heading", "section_level"
    ])
    dataset = dataset.rename_columns({
        "paragraph_text": "context",
        "paragraph_anchors": "anchors"
    })
    dataset = dataset.map(lambda x: prepare_features(x, tokenizer, max_length,
                                                     doc_stride, embeddings),
                          batched=True,
                          remove_columns=["context", "anchors"])
    dataset = dataset.filter(lambda x: len(x["spans"]) > 0)

    dataset = dataset.with_format(type="torch")

    return dataset, train_total, validation_total


def get_dataset_conll(tokenizer, embeddings):
    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train = load_dataset("cyanic-selkie/aida-conll-yago-wikidata",
                         split="train")
    train = train.shuffle(seed=42)
    train_total = len(train)

    validation = load_dataset("cyanic-selkie/aida-conll-yago-wikidata",
                              split="validation")
    validation_total = len(validation)

    dataset = IterableDatasetDict({
        "train":
        train.to_iterable_dataset(),
        "validation":
        validation.to_iterable_dataset()
    })

    dataset = dataset.remove_columns(["document_id"])
    dataset = dataset.rename_columns({
        "text": "context",
        "entities": "anchors"
    })
    dataset = dataset.map(lambda x: prepare_features(x, tokenizer, max_length,
                                                     doc_stride, embeddings),
                          batched=True,
                          remove_columns=["context", "anchors"])
    dataset = dataset.filter(lambda x: len(x["spans"]) > 0)

    dataset = dataset.with_format(type="torch")

    return dataset, train_total, validation_total


def get_dataset_cronel(tokenizer, embeddings):
    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train = load_dataset("cyanic-selkie/CroNEL", split="train")
    train = train.shuffle(seed=42)
    train_total = len(train)

    validation = load_dataset("cyanic-selkie/CroNEL", split="validation")
    validation_total = len(validation)

    dataset = IterableDatasetDict({
        "train":
        train.to_iterable_dataset(),
        "validation":
        validation.to_iterable_dataset()
    })

    dataset = dataset.remove_columns(["document_id"])
    dataset = dataset.rename_columns({
        "text": "context",
        "entities": "anchors"
    })
    dataset = dataset.map(lambda x: prepare_features(x, tokenizer, max_length,
                                                     doc_stride, embeddings),
                          batched=True,
                          remove_columns=["context", "anchors"])
    dataset = dataset.filter(lambda x: len(x["spans"]) > 0)

    dataset = dataset.with_format(type="torch")

    return dataset, train_total, validation_total


def get_dataset_cronel_conll(tokenizer, embeddings):
    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train_1 = load_dataset("cyanic-selkie/aida-conll-yago-wikidata",
                           split="train").remove_columns([
                               "document_id"
                           ]).shuffle(seed=42).rename_columns({
                               "text":
                               "context",
                               "entities":
                               "anchors"
                           })
    train_2 = load_dataset(
        "cyanic-selkie/CroNEL",
        split="train",
    ).remove_columns(["document_id"]).shuffle(seed=42).rename_columns({
        "text":
        "context",
        "entities":
        "anchors"
    })
    train_total = len(train_1) + len(train_2)

    validation_1 = load_dataset("cyanic-selkie/aida-conll-yago-wikidata",
                                split="validation").remove_columns([
                                    "document_id"
                                ]).shuffle(seed=42).rename_columns({
                                    "text":
                                    "context",
                                    "entities":
                                    "anchors"
                                })
    validation_2 = load_dataset(
        "cyanic-selkie/CroNEL",
        split="validation",
    ).remove_columns(["document_id"]).shuffle(seed=42).rename_columns({
        "text":
        "context",
        "entities":
        "anchors"
    })
    validation_total = len(validation_1) + len(validation_2)

    dataset_1 = IterableDatasetDict({
        "train":
        train_1.to_iterable_dataset(),
        "validation":
        validation_1.to_iterable_dataset()
    })
    dataset_2 = IterableDatasetDict({
        "train":
        train_2.to_iterable_dataset(),
        "validation":
        validation_2.to_iterable_dataset()
    })

    dataset_1 = dataset_1.map(lambda x: prepare_features(
        x, tokenizer, max_length, doc_stride, embeddings),
                              batched=True,
                              remove_columns=["context", "anchors"])
    dataset_2 = dataset_2.map(lambda x: prepare_features(
        x, tokenizer, max_length, doc_stride, embeddings),
                              batched=True,
                              remove_columns=["context", "anchors"])

    train = interleave_datasets(
        [dataset_1["train"], dataset_2["train"]],
        probabilities=[len(train_1) / train_total,
                       len(train_2) / train_total],
        seed=42)
    validation = interleave_datasets(
        [dataset_1["validation"], dataset_2["validation"]],
        probabilities=[
            len(validation_1) / validation_total,
            len(validation_2) / validation_total
        ],
        seed=42)

    dataset = IterableDatasetDict({"train": train, "validation": validation})

    dataset = dataset.filter(lambda x: len(x["spans"]) > 0)

    dataset = dataset.with_format(type="torch")

    return dataset, train_total, validation_total
