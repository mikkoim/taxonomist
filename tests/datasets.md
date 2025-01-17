

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

# test training with the webdataset

```bash
python scripts/02_train.py \
    --task "classification" \
    --dataset_config_path "conf/user_datasets.py" \
    --custom_dataset \
    --data_folder "data/raw/rodi/wds" \
    --dataset_name "rodi-wds" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'trivialaugment' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 10 \
    --min_epochs 0 \
    --early_stopping 'True' \
    --early_stopping_patience 10 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'False' \
    --log_dir 'roditest' \
    --out_folder 'outputs' \
    --out_prefix 'rodi' \
    --deterministic 'True'
```

