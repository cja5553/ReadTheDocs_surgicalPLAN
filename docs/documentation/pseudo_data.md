# Pseudo data

## `get_pseudo_data`

!!! note ""

    surgicalplan.**get_pseudo_data**()

Returns a fixed dataset of 500 hand-written preoperative clinical notes with hand-assigned binary outcomes, for testing and demonstration. The notes and labels are curated rather than generated: each note was written by hand, and each label assigned by reading that note. Labels correlate with clinical content, and include deliberately discordant cases, so a fine-tuned model learns probabilistic rather than deterministic associations.

!!! warning "Outcome prevalence is deliberately inflated"

    True postoperative DVT, pneumonia, and AKI incidence runs ~1–2%. At n=500 that would give roughly five positive cases per outcome — not enough for a model to learn from in a short demonstration run. Prevalence here sits near 19% for each outcome so that the signal is recoverable. **These are not epidemiological estimates and should not be read as risk figures.**

### Parameters

None.

### Returns

`pandas.DataFrame` with 500 rows and 5 columns:

- `clinical_note` (*str*) — hand-written preoperative note.
- `DVT`, `Pneumonia`, `AKI`, `Delirium` (*int*, 0/1) — hand-assigned outcomes.

### Example

```python
from surgicalplan import get_pseudo_data

df = get_pseudo_data()
print(df.shape)             # (500, 5)
print(df.columns.tolist())  # ['clinical_note', 'DVT', 'Pneumonia', 'AKI', 'Delirium']
```

See the [Examples](../examples.md) page for a full end-to-end demonstration using this dataset.
