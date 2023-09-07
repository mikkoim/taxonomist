```bash
# Load the data

a-get 2004353-puhti-SCRATCH/impiomik/data/finbenthic1.zip -d ./data/raw
unzip -q data/raw/finbenthic1.zip -d $TMPDIR


# Preprocessing

python scripts/preprocessing/process_finbenthic1.py \
    --folder "$TMPDIR/Detect dataset" \
    --out_folder "data/processed"

# train-test-split

python scripts/01_train_test_split.py \
    --csv_path "data/processed/finbenthic1-2/01_finbenthic1-2_processed.csv" \
    --target_col "taxon" \
    --group_col "individual" \
    --n_splits 5 \
    --out_folder "data/processed/finbenthic1-2"

# Train
sbatch batchjobs/train/finbenthic1/finbenthic1-2_efficientnet-b0.sh

# Predict
python scripts/03_predict.py \
    --data_folder "$TMPDIR/Detect dataset" \
    --dataset_name "finbenthic1" \
    --csv_path "data/processed/finbenthic1-2/01_finbenthic1-2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 0 \
    --class_map "data/processed/finbenthic1-2/label_map_finbenthic1-2.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'geometric' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix 'preds' \
    --ckpt_path "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11.ckpt"

python scripts/03_predict.py \
    --data_folder "$TMPDIR/Detect dataset" \
    --dataset_name "finbenthic1" \
    --csv_path "data/processed/finbenthic1-2/01_finbenthic1-2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 0 \
    --class_map "data/processed/finbenthic1-2/label_map_finbenthic1-2.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'geometric' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'True' \
    --out_prefix 'preds' \
    --ckpt_path "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11.ckpt"


## Grouping
python scripts/04_group_predictions.py \
    --predictions "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/predictions/geometric/preds_finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11_geometric.csv" \
    --reference_csv "data/processed/finbenthic1-2/01_finbenthic1-2_processed_5splits_taxon.csv" \
    --reference_target "taxon" \
    --fold 0 \
    --reference_group "individual" \
    --agg_func "mode"

python scripts/04_group_predictions.py \
    --predictions "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/predictions/geometric_tta/preds_finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11_geometric_tta.csv" \
    --reference_csv "data/processed/finbenthic1-2/01_finbenthic1-2_processed_5splits_taxon.csv" \
    --reference_target "taxon" \
    --fold 0 \
    --reference_group "individual" \
    --agg_func "mode"

## Evaluation

python scripts/05_evaluate.py \
    --predictions "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/predictions/geometric/preds_finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11_geometric.csv" \
    --metric_config conf/eval.yaml \
    --no_bootstrap

python scripts/05_evaluate.py \
    --predictions "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/predictions/geometric/preds_finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11_geometric_grouped.csv" \
    --metric_config conf/eval.yaml \
    --no_bootstrap

### TTA evaluation
python scripts/05_evaluate.py \
    --predictions "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/predictions/geometric_tta/preds_finbenthic1-2-base-200_efficientnet_b0_f0_230906-2108-e6de_epoch53_val-loss0.11_geometric_tta.csv" \
    --metric_config conf/eval.yaml \
    --no_bootstrap
```