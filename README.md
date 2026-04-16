# LLM Learning and Implementation (Hugging Face)

This repository contains implementations and experiments based on the Hugging Face course, covering key components of modern Large Language Models (LLMs), including tokenization, datasets, pipelines, and fine-tuning.

## Overview

- Implemented core LLM components from scratch and using Hugging Face
- Explored model pipelines, tokenizers (BPE), and dataset handling
- Built fine-tuning workflows with PyTorch and Hugging Face libraries
- Covered tokenization, training pipelines, and model adaptation for downstream tasks

## Structure

- Chpt2-pipelines/        # Model pipelines and inference
- Chpt3-fine-tuning/      # Fine-tuning (PyTorch, HF Trainer, Accelerate)
- Chpt5-datasets/         # Dataset loading and preprocessing
- Chpt6-tokenizers/       # Tokenization (BPE, normalization, fast tokenizers)

## Key Components

- **Fine-Tuning:** Implemented training pipelines using PyTorch and Hugging Face Trainer
- **Datasets:** Processed and loaded datasets using the `datasets` library
- **Tokenizers:** Implemented BPE and explored tokenizer design and preprocessing
- **Pipelines:** Used Hugging Face pipelines for inference

## Usage

```bash
pip install torch transformers datasets
