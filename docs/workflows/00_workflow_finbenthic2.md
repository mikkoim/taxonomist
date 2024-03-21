This workflow describes image classification for the FinBenthic2-dataset. It follows the same structure as the RODI dataset workflow, but with commands specific for the FinBenthic2-dataset.

## 1. Data loading

Load data like described in `docs/dataset_loading_finbenthic2.md`

## 2. Raw data analysis

Analyze the raw dataset like in `notebooks/01_raw_data_analysis_finbenthic2.ipynb`

## 3. Preprocess the data

```bash
python scripts/preprocessing/process_finbenthic2.py \
    --IDA_folder $TMPDIR/IDA \
    --out_folder data/processed/finbenthic2
```

## 4. Train-test-splits

```bash
python scripts/01_train_test_split.py \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed.csv" \
    --target_col "taxon" \
    --group_col "individual" \
    --n_splits 5 \
    --out_folder "data/processed/finbenthic2" \
    --random_state 123
```

We specified a random state `123` since the default resulted in a class missing from the first test fold. This is not necessary, but sometimes desired.

## 5. Training

If necessary, move files to a faster disk (HPC)

```bash
unzip -q data/raw/FIN-Benthic2.zip -d $TMPDIR
```

```bash
python scripts/02_train.py \
    --data_folder "$TMPDIR/IDA" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 0 \
    --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 2 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'cross-entropy' \
    --lr 0.0001 \
    --auto_lr 'True' \
    --log_dir 'finbenthic2test' \
    --out_folder 'outputs' \
    --out_prefix 'finbenthic2' \
    --deterministic 'True'
```

## 6. Prediction
Change the `ckpt_path` to the model in your outputs folder

```bash
python scripts/03_predict.py \
    --data_folder "$TMPDIR/IDA/" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 0 \
    --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
    --imsize 224 \
    --batch_size 128 \
    --aug 'none' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix 'finbenthic2' \
    --ckpt_path "outputs/finbenthic2/finbenthic2_resnet18_cross-entropy_b8/f0/finbenthic2_resnet18_cross-entropy_b8_f0_230831-2233-7ee4_epoch02_val-loss3.57.ckpt"
```

### 6.1 Prediction processing
Change the `predictions` to the predictions in your folder 
```bash
python scripts/04_group_predictions.py \
    --predictions "outputs/finbenthic2/finbenthic2_resnet18_cross-entropy_b8/f0/predictions/none/finbenthic2_finbenthic2_resnet18_cross-entropy_b8_f0_230831-2233-7ee4_epoch02_val-loss3.57_none.csv" \
    --reference_csv "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --reference_target "taxon" \
    --fold 0 \
    --reference_group "individual" \
    --agg_func "mode"
```

## 7. Evaluation

```bash
python scripts/05_evaluate.py \
    --predictions "outputs/finbenthic2/finbenthic2_resnet18_cross-entropy_b8/f0/predictions/none/finbenthic2_finbenthic2_resnet18_cross-entropy_b8_f0_230831-2233-7ee4_epoch02_val-loss3.57_none_grouped.csv" \
    --metric_config conf/eval.yaml
```

## 8. Comparison

```bash
python scripts/06_compare.py \
        --config 'conf/experiments/base_efficientnet.yaml' \
        --out_folder 'results'
```