# Synthetic Dataset

This directory contains the generated synthetic dataset used to train
and evaluate the Smart Drain ML models.

## Files

| File | Description |
|------|-------------|
| `generate_dataset.py` | Script to regenerate the dataset |
| `simulated_drainage_data.csv` | The generated CSV (created by running the script) |

## Regenerating

```bash
python data/generate_dataset.py
```

## Disclaimer

> Synthetic data are used for the present demonstration because a physical
> prototype and long-term field dataset are not available.  The data are
> generated using a simplified physics-inspired drainage model and do not
> represent actual field measurements.
