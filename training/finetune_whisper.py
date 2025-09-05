#!/usr/bin/env python
"""Minimal Whisper fine-tuning script.

This script fine-tunes an OpenAI Whisper model using Hugging Face
`transformers` and `datasets`. It expects manifests in JSON format where each
entry has an `audio` path and a `text` transcription.

The defaults target consumer GPUs like an RTX 4070 Ti (12 GB). Adjust batch
sizes or choose a smaller model if you encounter out-of-memory errors.

Example usage:

    python training/finetune_whisper.py train.json eval.json out_dir \
        --model_id openai/whisper-small --batch_size 2 --num_epochs 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Audio, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collate function that pads input features and labels."""

    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        label_batch = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt"
        )
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


def prepare_dataset(batch: Dict[str, Any], processor: WhisperProcessor) -> Dict[str, Any]:
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model.")
    parser.add_argument("train_manifest", help="JSON file with training data")
    parser.add_argument("eval_manifest", help="JSON file with evaluation data")
    parser.add_argument("output_dir", help="Directory to store checkpoints")
    parser.add_argument(
        "--model_id",
        default="openai/whisper-small",
        help="Model identifier on Hugging Face Hub",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    data_files = {"train": args.train_manifest, "validation": args.eval_manifest}
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=["audio", "text"],
    )

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
