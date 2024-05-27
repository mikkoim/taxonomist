```bash
gpushell 1 8G 32
module load allas
allas-conf

a-get 2004353-datasets/rodi_public.zip -d $TMPDIR
unzip $TMPDIR/rodi_public.zip -d $TMPDIR
source tykky

# Basic test
python scripts/02_train.py \
    --debug \
    --smoke_test \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_config "conf/user_datasets.py" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 8 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 2 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'cross-entropy' \
    --lr 0.004 \
    --auto_lr 'False' \
    --log_dir 'roditest_new' \
    --out_folder 'outputs' \
    --out_prefix 'rodi_new' \
    --deterministic 'False'

# Lr scheduler
python scripts/02_train.py \
    --smoke_test \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_config "conf/user_datasets.py" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 8 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 10 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'cross-entropy' \
    --lr 0.004 \
    --lr_scheduler 'CosineAnnealingLR' \
    --log_dir 'roditest_new' \
    --out_folder 'outputs' \
    --out_prefix 'rodi_new' \
    --deterministic 'False'
```