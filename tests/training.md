
# Normal training
This should train for around 23 epochs, or 5 minutes
```bash
python scripts/02_train.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 200 \
    --min_epochs 5 \
    --early_stopping 'True' \
    --early_stopping_patience 10 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'True' \
    --log_dir 'roditest' \
    --out_folder 'outputs' \
    --out_prefix 'rodi' \
    --deterministic 'True'
```

# No auto_lr
```bash
python scripts/02_train.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 5 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'False' \
    --log_dir 'roditest' \
    --out_folder 'outputs' \
    --out_prefix 'rodi-no-autolr' \
    --deterministic 'True'
```

# Continuing a failed run

```bash
python scripts/02_train.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 200 \
    --min_epochs 5 \
    --early_stopping 'True' \
    --early_stopping_patience 10 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'True' \
    --log_dir 'roditest' \
    --out_folder 'outputs' \
    --out_prefix 'rodi-continue' \
    --deterministic 'True'
    --ckpt_path <set this to something> \
    --resume 'True'
```