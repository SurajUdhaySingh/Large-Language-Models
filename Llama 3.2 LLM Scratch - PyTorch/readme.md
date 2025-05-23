# LLaMA-Based Language Model in PyTorch

This repository contains a minimalist implementation of a decoder-only transformer language model inspired by LLaMA, built from scratch using PyTorch. It focuses on core concepts such as rotary positional embeddings (RoPE), multi-head attention with grouped key-value heads, and RMS normalization.

## Features

- Decoder-only architecture
- Rotary positional embeddings (RoPE)
- Grouped key-value attention heads
- RMSNorm (Root Mean Square Layer Normalization)
- Modular PyTorch code structure
- Easily extensible for experimentation

## Model Architecture

The model consists of the following key components:

- **Config**: All hyperparameters are stored in a dataclass for easy configuration.
- **MultiHeadAttention**: Implements attention with RoPE and grouped KV heads.
- **LlamaMLP**: Position-wise feedforward network using gated activation (`SiLU`).
- **LlamaRMSNorm**: Custom implementation of RMS Layer Normalization.
- **LlamaBlock**: Combines attention and MLP with residual connections.
- **LlamaModel**: Stacks multiple transformer blocks and includes token embeddings.
- **LlamaLMHeadModel**: Final language model with optional label input for loss computation.

## Installation

Make sure you have PyTorch installed:

```bash
pip install torch
