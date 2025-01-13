

# Webdataset

Create webdataset from the RODI dataset:

```bash
python scripts/preprocessing/create_tardataset.py \
    --folder data/raw/rodi/Induced_Organism_Drift_2022 \
    --max_files 500 \
    --output_folder data/raw/rodi/wds \
    --output_prefix rodi \
    --shuffle


```