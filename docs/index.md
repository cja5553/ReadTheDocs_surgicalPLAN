# surgicalPLAN

**SurgicalPLAN** (*Surgical **P**ostoperative Risk Prediction with **L**anguage Models **A**dapting to Clinical **N**otes*) is a Python package for predicting postoperative risks from clinical notes using language models.

<p align="left">
  <a href="https://github.com/cja5553/ACS_demo_postoperative_risk_prediction_with_clinical_notes">
    <img src="https://img.shields.io/badge/Documentation-v0.1.0-006747" alt="Documentation">
  </a>
  <a href="https://pypi.org/project/surgicalplan/">
    <img src="https://img.shields.io/badge/pypi_package-v0.1.0-brightgreen" alt="pypi package">
  </a>
  <a href="https://github.com/cja5553/ACS_demo_postoperative_risk_prediction_with_clinical_notes">
    <img src="https://img.shields.io/badge/github_source_code-source_code?logo=github&color=BA0C2F" alt="GitHub Source Code">
  </a>
  <a href="https://www.nature.com/articles/s41746-025-01489-2">
    <img src="https://img.shields.io/badge/-npj_Digital_Medicine-grey?logo=nature&logoColor=white" alt="npj Digital Medicine">
  </a>
  <a href="https://github.com/cja5553/ACS_demo_postoperative_risk_prediction_with_clinical_notes/blob/main/license.txt">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </a>
</p>

## Overview

SurgicalPLAN provides flexible and clinically oriented workflows that support a range of perioperative use cases, enabling clinicians, researchers, and healthcare institutions to train and fine-tune models using preoperative or intraoperative clinical text.

The package is designed to be accessible to a broad range of users, including clinicians, surgeons, and researchers with limited programming experience. It minimizes the need to interact with lower-level machine learning frameworks such as PyTorch. With just a few lines of high-level functions, users can begin training and fine-tuning their own models.

SurgicalPLAN supports multiple modeling strategies:

1. **Direct inference** with fine-tuned language models — score clinical text against a ready-made model with no training step. See [Direct inference](documentation/direct_inference.md).
2. **(Joint) semi-supervised learning** for leveraging partially labeled data, trained for a single outcome at a time. See [Joint finetuning](documentation/joint_finetuning.md).
3. A **multi-task learning framework** that enables simultaneous prediction of multiple postoperative outcomes from the same notes. See [Multi-task finetuning](documentation/multitask_finetuning.md).

The package was developed for the American College of Surgeons (ACS) workshop, *AI for Clinicians and Surgeons: A Hands-On Introduction Across the Care Continuum*.

!!! note "Which workflow should I use?"

    - **No labeled data, want results now?** Use [`direct_inference_from_trained_model`](documentation/direct_inference.md) — it loads a pre-trained model from HuggingFace Hub and scores notes immediately.
    - **One outcome of interest?** Use [`joint_finetune`](documentation/joint_finetuning.md) to train a model specialized for a single outcome.
    - **Several outcomes at once?** Use [`mtl_finetune`](documentation/multitask_finetuning.md) to train one model that predicts multiple risks simultaneously.

## Requirements

### Python version

Python **3.9–3.12** (tested on 3.12).

### Required packages

`pip install surgicalplan` resolves all of the following automatically. They are listed here for reference:

- `transformers` (>=4.36, <5)
- `tokenizers` (>=0.15)
- `huggingface_hub` (>=0.20)
- `accelerate` (>=0.25)
- `datasets` (>=2.14)
- `safetensors` (>=0.4)
- `torch` (>=2.0)
- `numpy` (>=1.23)
- `pandas` (>=2.0)
- `pyarrow` (>=14)
- `tqdm` (>=4.66)

**Optional extras:**

- `classifiers` — adds `scikit-learn` (>=1.3) and `xgboost` (>=1.7), for downstream classifier workflows. Install with `pip install "surgicalplan[classifiers]"`.
- `dev` — adds `pytest`, `jupyter`, `ipykernel`, and `ipywidgets`, for development and notebooks. Install with `pip install "surgicalplan[dev]"`.

## Installation

```bash
pip install surgicalplan
```

!!! warning "Install PyTorch first"

    Because `torch` CUDA wheels aren't hosted on PyPI, install PyTorch first matching your GPU's CUDA version, then install this package. For example, on a machine with CUDA 11.8 drivers:

    ```bash
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install surgicalplan
    ```

## Quick example

The package ships with [`get_pseudo_data`](documentation/multitask_finetuning.md#get_pseudo_data), a synthetic dataset generator, so you can run an end-to-end example without any private clinical data.

```python
from surgicalplan import (
    get_pseudo_data,
    mtl_finetune,
    get_postoperative_outcome_scores,
)

# 1. Get a small synthetic dataset for demonstration
df = get_pseudo_data()
# df columns: "text", "Outcome_1", "Outcome_2", "Outcome_3", "Outcome_4"

# 2. Fine-tune a multi-task model across all four outcomes
mtl_finetune(
    df,
    text_col="text",
    outcome_cols=["Outcome_1", "Outcome_2", "Outcome_3", "Outcome_4"],
    output_dir="my_finetuned_model",
)

# 3. Score a new clinical scenario
note = (
    "83-year-old male, ASA 4, scheduled for coronary artery bypass graft "
    "(emergent three-vessel). Indication: severe CAD with LAD stenosis, "
    "presenting with unstable angina. PMH: COPD, type 2 diabetes mellitus, "
    "coronary artery disease, prior MI, chronic kidney disease stage 3. "
    "Social: current smoker, 1 pack per day. BMI 34 (obese). Allergies: NKDA."
)

scores = get_postoperative_outcome_scores("my_finetuned_model", note)
print(scores)
# {'Outcome_1': 0.12, 'Outcome_2': 0.28, 'Outcome_3': 0.04, 'Outcome_4': 0.39}
```

For details on each function and its parameters, refer to the [Documentation](documentation/index.md).

## Citation

If you use this package, please cite the accompanying work:

> Alba, C., Xue, B., Abraham, J., Kannampallil, T., & Lu, C. "The foundational capabilities of large language models in predicting postoperative risks using clinical notes." *npj Digital Medicine* 8, 95 (2025). doi: [10.1038/s41746-025-01489-2](https://www.nature.com/articles/s41746-025-01489-2)

## Questions?

Contact me at [alba@wustl.edu](mailto:alba@wustl.edu)
