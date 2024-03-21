#!/bin/bash
#SBATCH --job-name=eb0
#SBATCH --account=Project_2004353
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb0.txt"
#SBATCH -e "e_eb0.txt"

# This batchjob trains an initial model from scratch

echo "Extracting data..."
unzip -q data/raw/FIN-Benthic2.zip -d $TMPDIR
echo "Done!"
source tykky
srun python scripts/02_train.py \
                --data_folder "$TMPDIR/IDA/" \
                --dataset_name "finbenthic2" \
                --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
                --label "taxon" \
                --fold 1 \
                --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
                --imsize 224 \
                --batch_size 256 \
                --aug 'aug-02' \
                --load_to_memory 'False' \
                --model 'efficientnet_b0' \
                --freeze_base 'False' \
                --pretrained 'True' \
                --opt 'adamw' \
                --max_epochs 200 \
                --min_epochs 5 \
                --early_stopping 'False' \
                --early_stopping_patience 50 \
                --criterion 'cross-entropy' \
                --lr 0.0001 \
                --auto_lr 'True' \
                --log_dir 'benthic-models' \
                --out_folder 'outputs' \
                --out_prefix 'finbenthic2-base-200' \
                --deterministic 'True'