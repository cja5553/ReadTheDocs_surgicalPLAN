# Direct inference

Direct inference lets you use out-of-the-box models that have already been trained on clinical data and its associated postoperative outcomes. Unlike the finetuning workflows, this is a direct inference function that loads a pre-trained, ready-to-use model from HuggingFace Hub and therefore requires **no model training**.

The default model is [`cja5553/BJH-perioperative-notes-bioClinicalBERT`](https://huggingface.co/cja5553/BJH-perioperative-notes-bioClinicalBERT), a Bio+ClinicalBERT variant that was multi-task fine-tuned across six postoperative outcomes: (1) death within 30 days, (2) DVT, (3) PE, (4) AKI, (5) delirium, and (6) pneumonia. This model was used in the accompanying [*npj Digital Medicine* paper](https://www.nature.com/articles/s41746-025-01489-2).

## `direct_inference_from_trained_model`

!!! note ""

    surgicalplan.**direct_inference_from_trained_model**(*text, outcomes=None, model_name="cja5553/BJH-perioperative-notes-bioClinicalBERT", max_length=None, device=None, hf_token=None*)

Score clinical text against a pre-trained multi-task model without any fine-tuning step. The model is downloaded from HuggingFace Hub on first use and cached locally thereafter.

### Parameters

- `text` (*str | list[str]*, **required**): One clinical scenario, or a list of them. Determines the shape of the return value.
- `outcomes` (*list[str] | None*, optional): Which outcomes to score. Defaults to all outcomes the default model was trained on (`DVT`, `PE`, `PNA`, `postop_del`, `death_in_30`, `post_aki_status`), recovered from the model's `mtl_metadata.json`. Pass a subset to score only some.
- `model_name` (*str*, optional): HuggingFace repo ID or local path. Override to use your own fine-tuned model. Defaults to `"cja5553/BJH-perioperative-notes-bioClinicalBERT"`.
- `max_length` (*int | None*, optional): Token sequence length. Defaults to the value used during fine-tuning, recovered from metadata.
- `device` (*str | None*, optional): `"cuda"`, `"cpu"`, or `None` to auto-detect.
- `hf_token` (*str | None*, optional): Optional HuggingFace token, required only if the model repo is gated/private.

### Returns

- `dict[str, float]` when `text` is a string — maps each outcome name to a probability in `[0, 1]`.
- `list[dict[str, float]]` when `text` is a list — one dict per input, in the same order.

### Example

```python
from surgicalplan import direct_inference_from_trained_model

note = (
    "Redo coronary artery bypass graft with aortic valve replacement "
    "bioprosthetic. Indication: severe ischemic cardiomyopathy, "
    "ejection fraction 25 percent, prior MI, ventricular arrhythmia "
    "status post AICD placement, stage 3 chronic kidney disease, COPD."
)

scores = direct_inference_from_trained_model(text=note)
print(scores)
# {'DVT': 0.17, 'PE': 0.06, 'PNA': 0.28, 'postop_del': 0.81,
#  'death_in_30': 0.46, 'post_aki_status': 0.93}
```

To score only a subset of outcomes, pass `outcomes`:

```python
scores = direct_inference_from_trained_model(
    text=note,
    outcomes=["death_in_30", "post_aki_status"],
)
# {'death_in_30': 0.46, 'post_aki_status': 0.93}
```

You can also score a batch of notes by passing a list:

```python
notes = [note_a, note_b, note_c]
all_scores = direct_inference_from_trained_model(text=notes)
# [ {...}, {...}, {...} ]  one dict per note, in order
```

!!! tip "Notes"

    - The first call downloads the model (~440 MB) from HuggingFace and caches it locally; subsequent calls use the cache.
    - Inference runs on CPU in ~5 seconds per note, or ~0.5 seconds with a GPU.
    - To use your own fine-tuned model instead of the default, pass its directory or HuggingFace repo as `model_name`.

For users who want to fine-tune their own model, see [`mtl_finetune`](multitask_finetuning.md) (multi-outcome) or [`joint_finetune`](joint_finetuning.md) (single-outcome).
