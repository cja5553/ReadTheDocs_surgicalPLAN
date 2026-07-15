# Examples

This page walks through a complete, runnable example using the synthetic dataset shipped with the package, so you can try the full pipeline without any private clinical data.

## End-to-end multi-task example

The following demonstrates the full multi-task workflow: load data → fine-tune → score a new scenario.

```python
from surgicalplan import (
    get_pseudo_data,
    mtl_finetune,
    get_postoperative_outcome_scores,
)

# 1. Load the bundled dataset (500 hand-written notes, 4 binary outcomes)
df = get_pseudo_data()
print(df.shape)             # (500, 5)
print(df.columns.tolist())  # ['clinical_note', 'DVT', 'Pneumonia', 'AKI', 'Delirium']

# 2. Fine-tune a multi-task model across all four outcomes
mtl_finetune(
    df,
    text_col="clinical_note",
    outcome_cols=["DVT", "Pneumonia", "AKI", "Delirium"],
    output_dir="demo_mtl_model",
    training_configs={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
    },
)

# 3. Score a new clinical scenario
note = (
    "74M, ASA 4, for coronary artery bypass grafting x3 with aortic valve "
    "replacement. Indication: three-vessel disease with severe aortic stenosis, "
    "EF 35%. PMH: CKD stage 3 Cr 1.8, T2DM on insulin, HTN."
)

scores = get_postoperative_outcome_scores("demo_mtl_model", note)
print(scores)
# {'DVT': 0.31, 'Pneumonia': 0.24, 'AKI': 0.55, 'Delirium': 0.52}
```

## Single-outcome (joint) example

If you care about just one outcome, train a joint single-outcome model instead:

```python
from surgicalplan import get_pseudo_data, joint_finetune, get_outcome_score

df = get_pseudo_data()

# Train a model specialized for DVT
joint_finetune(
    df,
    text_col="clinical_note",
    outcome_col="DVT",
    output_dir="demo_joint_model",
    training_configs={"num_train_epochs": 3, "learning_rate": 2e-5},
)

prob = get_outcome_score(model_name="demo_joint_model", text=note)
print(prob)
# 0.18
```

## No-training example (direct inference)

To skip training entirely and use the pre-trained model from the [*npj Digital Medicine* paper](https://www.nature.com/articles/s41746-025-01489-2):

```python
from surgicalplan import direct_inference_from_trained_model

scores = direct_inference_from_trained_model(text=note)
print(scores)
# {'DVT': 0.17, 'PE': 0.06, 'PNA': 0.28, 'postop_del': 0.81,
#  'death_in_30': 0.46, 'post_aki_status': 0.93}
```

---

For full parameter documentation on each function, see the [Documentation](documentation/index.md) section.
