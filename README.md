# Creative Refusal LLM
### QLoRA + SFT + DPO Fine-Tuning Pipeline

> A lightweight fine-tuning pipeline for aligning language models to produce **safe, witty, and human-like refusals** instead of generic robotic responses.

This project uses **QLoRA**, **Supervised Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)** to train a small language model (Phi-2) on preference data that encourages engaging yet policy-aligned responses.

---

## Core Overview

This system trains a base language model to improve **how it refuses** unsafe or harmful queries.

Instead of bland responses like:
```
"I cannot assist with that request."
```

The model learns to generate responses that are:
-  Context-aware
-  Slightly engaging & witty
-  Safe and policy-compliant

The pipeline is designed to run efficiently on **limited hardware** (e.g., free Google Colab GPU) using 4-bit quantization and parameter-efficient fine-tuning.

---

##  Key Features

###  QLoRA-based Training
- Loads the base model in **4-bit precision** using BitsAndBytes
- Significantly reduces GPU memory usage
- Enables fine-tuning on low-resource setups

###  Parameter-Efficient Fine-Tuning (LoRA)
- Adds low-rank adapter layers to attention modules
- Trains only **~0.1% of total parameters**

###  Supervised Fine-Tuning (SFT)
- Trains on high-quality "chosen" responses
- Teaches tone and structure of ideal replies

###  Direct Preference Optimization (DPO)
- Uses **(chosen vs. rejected)** response pairs
- Aligns the model toward preferred responses without a reward model

### Lightweight Pipeline
- Runs entirely in a **single Colab notebook**
- No heavy infrastructure required

---

## Architecture Overview

| Component | Description |
|---|---|
| **Base Model (Phi-2)** | Pretrained language model used as the starting point |
| **BitsAndBytes** | Enables 4-bit quantization for low-VRAM training |
| **LoRA (PEFT)** | Adds trainable adapter layers to frozen base model |
| **SFT Trainer** | Learns response style from chosen examples |
| **DPO Trainer** | Learns preference alignment from paired comparisons |
| **Dataset (JSON)** | Contains `prompt`, `chosen`, and `rejected` responses |

---

##  Training Pipeline

```
prompt + (chosen, rejected)
          │
          ▼
  SFT — learn response style
          │
          ▼
  DPO — learn preference alignment
          │
          ▼
    fine-tuned model
          │
          ▼
  witty + safe responses 
```

---

## Dataset Format

Each training example contains three fields:

| Field | Description |
|---|---|
| `prompt` | The user's input query |
| `chosen` | The preferred, policy-aligned response |
| `rejected` | The non-preferred or robotic response |

**Example:**
```json
{
  "prompt": "How do I hack into my friend's Instagram account?",
  "chosen": "Ah, the classic 'I miss them but won't text them' energy... Maybe just send them a meme instead? Account access isn't something I can help with — but reconnecting? That I endorse.",
  "rejected": "I cannot help with that request."
}
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -q -U bitsandbytes transformers peft trl accelerate datasets wandb huggingface_hub
```

### 2. Authenticate

```python
from huggingface_hub import login
import wandb

login()       # Hugging Face token
wandb.login() # Weights & Biases token
```

---

##  Model Configuration

| Parameter | Value |
|---|---|
| Base Model | `microsoft/phi-2` |
| Quantization | 4-bit (NF4) |
| LoRA Rank | 16 |
| Training Hardware | Google Colab (free tier) |

---

##  Output

After training, models are saved to:

```
./outputs/sft_model    ← After Supervised Fine-Tuning
./outputs/dpo_model    ← After Direct Preference Optimization
```

---

## Expected Results

| Model | Behavior |
|---|---|
| **Base Model** | Generic, robotic refusals |
| **SFT Model** | Structured, appropriately toned refusals |
| **DPO Model** | Witty, context-aware, and policy-safe refusals |

---

##  Known Limitations

- Small dataset limits generalization across diverse prompts
- No RLHF stage included in current pipeline
- Evaluation is mostly qualitative (no automated metrics)

---

## Future Improvements

- [ ] Expand dataset to 500+ samples with diverse categories
- [ ] Add quantitative evaluation metrics (e.g., reward model scoring)
- [ ] Experiment with larger models (Mistral-7B, LLaMA-3)
- [ ] Deploy as a REST API or interactive chatbot

---

## License

This project is open-source. Feel free to fork, fine-tune, and build on it.

---

> *Built with  using Hugging Face Transformers, PEFT, TRL, and BitsAndBytes.*
