# Normal training
This should train for around 23 epochs, or 5 minutes
```bash
python scripts/02_train.py \
    --no_wandb \
    --task "classification" \
    --dataset_config_path "conf/user_datasets.py" \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
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
    --max_epochs 2 \
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

# Continuing a failed run

```bash
python scripts/02_train.py \
    --no_wandb \
    --task "classification" \
    --dataset_config_path "conf/user_datasets.py" \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
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
    --max_epochs 5 \
    --min_epochs 0 \
    --early_stopping 'True' \
    --early_stopping_patience 10 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'False' \
    --log_dir 'roditest' \
    --out_folder 'outputs' \
    --out_prefix 'rodi' \
    --deterministic 'True' \
    --ckpt_path "outputs/rodi/rodi_resnet18/f0/rodi_resnet18_f0_250116-1641-ed84_epoch01_val-loss1.77_last.ckpt" \
    --resume 'True'
```

# prediction

```bash
python scripts/03_predict.py \
    --task "classification" \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --dataset_config "conf/user_datasets.py" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --ckpt_path "outputs/rodi/rodi_resnet18/f0/rodi_resnet18_f0_250116-1649-e414_epoch01_val-loss1.77_last.ckpt"

python scripts/03_predict.py \
    --task "classification" \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --dataset_config "conf/user_datasets.py" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'True' \
    --out_prefix '' \
    --ckpt_path "outputs/rodi/rodi_resnet18/f0/rodi_resnet18_f0_250116-1649-e414_epoch01_val-loss1.77_last.ckpt"

# Features
python scripts/03_predict.py \
    --task "feature-extraction" \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --dataset_config "conf/user_datasets.py" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --ckpt_path "outputs/rodi/rodi_resnet18/f0/rodi_resnet18_f0_250116-1649-e414_epoch01_val-loss1.77_last.ckpt" \
    --feature_extraction "pooled"

# features without checkpoint
python scripts/03_predict.py \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --dataset_config "conf/user_datasets.py" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --timm_model_name 'mobilenetv3_large_100.ra_in1k' \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --feature_extraction "pooled"
```