#!/bin/bash
#SBATCH --job-name=base
#SBATCH --account=Project_2004353
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_base.txt"
#SBATCH -e "e_base.txt"

# This batchjob trains an initial model from scratch

echo "Extracting data..."
unzip -q data/raw/FIN-Benthic2.zip -d $TMPDIR
echo "Done!"
source tykky
srun python scripts/02_train.py \
                --data_folder "$TMPDIR/IDA/" \
                --dataset_config "conf/user_datasets.py" \
                --dataset_name "finbenthic2" \
                --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
                --label "taxon" \
                --fold 0 \
                --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
                --imsize 224 \
                --batch_size 256 \
                --aug 'aug-02' \
                --load_to_memory 'False' \
                --model 'efficientnet_b0' \
                --freeze_base 'False' \
                --pretrained 'True' \
                --opt 'adamw' \
                --lr_scheduler 'CosineAnnealingLR' \
                --max_epochs 200 \
                --min_epochs 5 \
                --early_stopping 'False' \
                --criterion 'cross-entropy' \
                --lr 0.0001 \
                --auto_lr 'True' \
                --log_dir 'benchmarks' \
                --out_folder 'outputs' \
                --out_prefix 'finbenthic2-cosine' \
                --deterministic 'True'