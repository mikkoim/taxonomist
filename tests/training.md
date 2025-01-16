# Normal training
```bash
python scripts/02_train.py \
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

# Continuing a run
```bash
python scripts/02_train.py \
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
    --ckpt_path "outputs/rodi/rodi_resnet18/f0/rodi_resnet18_f0_250116-1817-437c_epoch01_val-loss1.77_last.ckpt" \
    --resume 'True'
```

# Regression
```bash
python scripts/02_train.py \
    --no_wandb \
    --task "regression" \
    --dataset_config_path "conf/user_datasets.py" \
    --data_folder "data/raw/biomass/biomass_subset/images" \
    --dataset_name "biomass" \
    --csv_path "data/processed/biomass/maamet_processed_asaq_5splits_weight_log.csv" \
    --label "weight_log" \
    --fold 0 \
    --class_map "none" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'flips-rotate' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 3 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'l1' \
    --lr 0.001 \
    --auto_lr 'False' \
    --log_dir 'biomasstest' \
    --out_folder 'outputs' \
    --out_prefix 'biomass' \
    --deterministic 'True'
```

# using pretrained weights
```bash
python scripts/02_train.py \
    --no_wandb \
    --task "regression" \
    --dataset_config_path "conf/user_datasets.py" \
    --data_folder "data/raw/biomass/biomass_subset/images" \
    --dataset_name "biomass" \
    --csv_path "data/processed/biomass/maamet_processed_asaq_5splits_weight_log.csv" \
    --label "weight_log" \
    --fold 0 \
    --class_map "none" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'flips-rotate' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 3 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'l1' \
    --lr 0.001 \
    --auto_lr 'False' \
    --log_dir 'biomasstest' \
    --out_folder 'outputs' \
    --out_prefix 'biomass' \
    --deterministic 'True' \
    --ckpt_path "outputs/rodi/rodi-new_resnet18/f0/rodi-new_resnet18_f0_250116-1821-f073_epoch02_val-loss1.02_last.ckpt"
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
    --ckpt_path "outputs/rodi/rodi-new_resnet18/f0/rodi-new_resnet18_f0_250116-1821-f073_epoch02_val-loss1.02_last.ckpt"

# regression
python scripts/03_predict.py \
    --no_wandb \
    --task "regression" \
    --dataset_config_path "conf/user_datasets.py" \
    --data_folder "data/raw/biomass/biomass_subset/images" \
    --dataset_name "biomass" \
    --csv_path "data/processed/biomass/maamet_processed_asaq_5splits_weight_log.csv" \
    --label "weight_log" \
    --fold 0 \
    --class_map "none" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --ckpt_path "outputs/biomass/biomass_resnet18/f0/biomass_resnet18_f0_250116-1841-e4d0_epoch02_val-loss0.46_last.ckpt"

# TTA true
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
    --ckpt_path "outputs/rodi/rodi-new_resnet18/f0/rodi-new_resnet18_f0_250116-1821-f073_epoch02_val-loss1.02_last.ckpt"

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
    --ckpt_path "outputs/rodi/rodi-new_resnet18/f0/rodi-new_resnet18_f0_250116-1821-f073_epoch02_val-loss1.02_last.ckpt" \
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