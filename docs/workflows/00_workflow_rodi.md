This workflow describes image classification for the RODI dataset (de Schaetzen et al. 2023)

## 1. Data loading

Load data like described in `docs/dataset_loading.md`

## 2. Raw data analysis

Analyze the raw dataset like in `notebooks/01_raw_data_analysis_rodi.ipynb`

## 3. Preprocess the data

```powershell
python scripts\\preprocessing\\process_rodi.py `
    --csv_path="data\\raw\\rodi\\Induced_Organism_Drift_2022_annotations.csv" `
    --out_folder="data\\processed\\rodi"
```

```bash
python scripts/preprocessing/process_rodi.py \
    --csv_path="data/raw/rodi/Induced_Organism_Drift_2022_annotations.csv" \
    --out_folder="data/processed/rodi"
```

## 4. Train-test-splits

```powershell
python scripts/01_train_test_split.py `
    --csv_path "data\\processed\\rodi\\01_rodi_processed.csv" `
    --target_col "family" `
    --group_col "ind_id" `
    --n_splits 5 `
    --out_folder "data\\processed\\rodi"
```

```bash
python scripts/01_train_test_split.py \
    --csv_path "data/processed/rodi/01_rodi_processed.csv" \
    --target_col "family" \
    --group_col "ind_id" \
    --n_splits 5 \
    --out_folder "data/processed/rodi"
```

## 5. Training
```powershell
python -m pdb scripts\\02_train.py `
    --data_folder "data\\raw\\rodi\\Induced_Organism_Drift_2022" `
    --dataset_name "rodi" `
    --csv_path "data\\processed\\rodi\\01_rodi_processed_5splits_family.csv" `
    --label "family" `
    --fold 0 `
    --class_map "data\\processed\\rodi\\rodi_label_map.txt" `
    --imsize 224 `
    --batch_size 256 `
    --aug 'aug-02' `
    --load_to_memory 'False' `
    --model 'resnet18' `
    --opt 'adamw' `
    --max_epochs 200 `
    --min_epochs 5 `
    --early_stopping 'True' `
    --early_stopping_patience 10 `
    --criterion 'cross-entropy' `
    --lr 0.0001 `
    --auto_lr 'True' `
    --log_dir 'rodi' `
    --out_folder 'outputs' `
    --out_prefix 'rodi' `
    --deterministic 'True'
```
```bash
python scripts/02_train.py \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 200 \
    --min_epochs 5 \
    --early_stopping 'True' \
    --early_stopping_patience 10 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'True' \
    --log_dir 'rodi' \
    --out_folder 'outputs' \
    --out_prefix 'rodi' \
    --deterministic 'True'
```

## 6. Predicting
python scripts/03_predict.py \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'True' \
    --out_prefix 'preds' \
    --ckpt_path "outputs/rodi/rodi_resnet18/f0/rodi_resnet18_f0_240322-1506-12f9_epoch14_val-loss0.26.ckpt"