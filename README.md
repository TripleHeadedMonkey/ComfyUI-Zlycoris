# Z-Image Toolkit (LyCORIS / LoRA / GGUF / Merge Tools)

Advanced raw patching toolkit for ComfyUI.

## Features

- LyCORIS / LoHA / LoKR Loader
- AITK LoRA Loader (with stacking)
- GGUF Raw Loader & Injector
- Vector Merge
- TIES Merge
- Raw Model Merge
- Diffusers Loader
- CLIP & Model Inject / Uninject
- Utility Nodes

## Categories

- Z-Image/Loaders
- Z-Image/Injectors
- Z-Image/Saving
- Experimental
- utils

## Installation

Install via ComfyUI Manager or clone manually:


cd ComfyUI/custom_nodes

git clone https://github.com/TripleHeadedMonkey/ComfyUI-Zlycoris.git

Restart ComfyUI.

## more info

Zlycoris – ComfyUI Nodes for Transformer Merging & LoRA Automation

Zlycoris is a collection of ComfyUI nodes focused on two main areas:

Transformer model merging (including Qwen 3 4B support)

Advanced LoRA loading and prompt-driven automation utilities

This extension is designed to give you structured control over model merging and dynamic LoRA behavior inside ComfyUI workflows.

Features
🧠 Transformer Model Merging

Includes nodes for merging full transformer models using structured weight merging techniques.

Highlights

TIES-based transformer merging

Qwen 3 4B-specific merging nodes

GGUF raw loading support

Dequantization utilities

Advanced weight operations

Use Case

Blend multiple Qwen models into a hybrid model

Combine instruction-tuned and creative variants

Experiment with task arithmetic on transformer checkpoints

Create custom merged models inside ComfyUI

These nodes operate at the full model weight level — not LoRA merging.

🎛 Universal LoRA / LyCORIS Loader

Supports multiple LoRA-style formats through a unified loader.

Supported Types

LoRA

LyCORIS variants

LoHa

LoKr

DoRA (if applicable)

Use Case

Load LoRAs directly into your diffusion pipeline

Adjust strength dynamically

Stack multiple adapters

Build structured LoRA systems

🔍 Keyword Matching Gate

A utility node that detects keywords in prompts and outputs boolean signals.

Use Case

Enable specific LoRAs only when keywords appear

Automatically activate lighting, style, or character adapters

Build smart prompt-reactive workflows

Example:

If prompt contains "cinematic" → enable lighting LoRA

If prompt contains "anime" → disable photoreal LoRA

🔄 Primitive to String Utilities

Converts numeric or widget values into strings for dynamic parameter control.

Use Case

Dynamically control LoRA strength

Build string-based automation systems

Drive behavior from sliders or external values

Useful when combining user inputs with prompt-aware logic.

⚙️ Conditional & Advanced Conditioning Nodes

Includes tools for:

Conditional routing

Conditioning scaling (e.g., CondMul)

Advanced conditioning manipulation (ZCondAdv)

Use Case

Multiply or scale conditioning signals

Create branch-based LoRA stacking systems

Build structured style control systems

📦 GGUF & Dequant Utilities

Utilities for:

Loading raw GGUF transformer models

Dequantizing weights

Preparing models for merging

Use Case

Work with quantized Qwen models

Experiment with merging lower-memory checkpoints

Prototype transformer blends inside ComfyUI

Example Workflow Concepts

These nodes are modular and designed to work together. Here are simple examples of how they might be used:

1️⃣ Prompt-Driven LoRA Activation

Feed prompt into Keyword Match Gate

Use output to enable/disable specific LoRAs

Adjust strength with slider or numeric input

Result: One workflow that automatically adapts to prompt content.

2️⃣ Dynamic LoRA Strength Control

Use slider → Primitive to String

Combine with keyword detection

Adjust LoRA intensity based on words like:

“slightly”

“very”

“extremely”

Result: Prompt-aware LoRA strength scaling.

3️⃣ Transformer Personality Mixer (Qwen)

Load multiple Qwen models

Merge with TIES node

Adjust merge ratios with sliders

Result: Custom blended LLM personality or task behavior.

4️⃣ Structured LoRA Stacking System

Create separate branches:

Character LoRAs

Lighting LoRAs

Texture LoRAs

Style LoRAs

Control each branch with:

Keyword gates

Sliders

Conditional routing

Result: Clean, modular LoRA architecture instead of uncontrolled stacking.

Design Philosophy

Zlycoris focuses on:

Structured model merging

Modular LoRA control

Prompt-aware automation

Workflow-driven experimentation

It is designed for users who want more control than simple “load LoRA and set strength” systems.

Intended Users

This extension is best suited for:

Advanced ComfyUI users

Model experimenters

Users blending Qwen models

Creators building automated prompt-reactive pipelines
