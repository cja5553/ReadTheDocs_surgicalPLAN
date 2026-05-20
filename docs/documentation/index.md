# Documentation

SurgicalPLAN exposes three workflows for predicting postoperative risks from clinical notes. They differ in whether you train a model and how many outcomes you predict. All public functions are importable directly from the top-level package:

```python
from surgicalplan import (
    direct_inference_from_trained_model,   # Direct inference
    joint_finetune, get_outcome_score,     # Joint finetuning
    mtl_finetune, get_postoperative_outcome_scores, get_pseudo_data,  # Multi-task
)
```

## The three workflows

| Workflow | Training required? | Outcomes | Use when |
| --- | --- | --- | --- |
| [Direct inference](direct_inference.md) | No | Multiple (from a pre-trained model) | You want results immediately with no labeled data |
| [Joint finetuning](joint_finetuning.md) | Yes | One | You care about a single outcome and want a specialized model |
| [Multi-task finetuning](multitask_finetuning.md) | Yes | Many | You want one model that predicts several outcomes at once |

### 1. Direct inference

Load a pre-trained, ready-to-use model from HuggingFace Hub and score clinical text with no fine-tuning step. The default model was multi-task fine-tuned across six postoperative outcomes. Start here if you have no labeled data of your own.

→ [`direct_inference_from_trained_model`](direct_inference.md)

### 2. Joint (semi-supervised) finetuning

Train a separate model for each outcome of interest. The model jointly learns the structure of your clinical notes (via masked language modeling) while learning to predict the outcome, capturing both your institution's documentation style and the clinical features that drive your specific outcome.

→ [`joint_finetune`](joint_finetuning.md), [`get_outcome_score`](joint_finetuning.md#get_outcome_score)

### 3. Multi-task finetuning

Train a single versatile model capable of predicting multiple postoperative outcomes from the same clinical notes — analogous to a foundation model. Instead of one model per outcome, you get one model with one classification head per outcome.

→ [`mtl_finetune`](multitask_finetuning.md), [`get_postoperative_outcome_scores`](multitask_finetuning.md#get_postoperative_outcome_scores), [`get_pseudo_data`](multitask_finetuning.md#get_pseudo_data)

## A note on saved models

Both `joint_finetune` and `mtl_finetune` save everything needed for inference into the `output_dir` you specify — model weights, tokenizer, and a metadata JSON (`joint_metadata.json` or `mtl_metadata.json`). The corresponding scoring functions read that metadata automatically, so you don't have to re-specify `max_length`, outcome names, or the base model at inference time.
