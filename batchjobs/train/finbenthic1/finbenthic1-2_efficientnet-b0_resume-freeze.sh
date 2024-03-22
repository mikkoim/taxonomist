#!/bin/bash
#SBATCH --job-name=eb0-frz
#SBATCH --account=Project_2004353
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb0-frz.txt"
#SBATCH -e "e_eb0-frz.txt"

# This batchjob resumes a trained model with frozen convolutional layers

echo "Extracting data..."
unzip -q data/raw/finbenthic1.zip -d $TMPDIR
echo "Done!"
source tykky
srun python scripts/02_train.py \
                --data_folder "$TMPDIR/Detect dataset/" \
                --dataset_name "finbenthic1" \
                --csv_path "data/processed/finbenthic1-2/01_finbenthic1-2_processed_5splits_taxon.csv" \
                --label "taxon" \
                --fold 0 \
                --class_map "data/processed/finbenthic1-2/label_map_finbenthic1-2.txt" \
                --imsize 224 \
                --batch_size 256 \
                --aug 'aug-02' \
                --load_to_memory 'False' \
                --model 'efficientnet_b0' \
                --freeze_base 'True' \
                --pretrained 'True' \
                --opt 'adamw' \
                --max_epochs 200 \
                --min_epochs 5 \
                --early_stopping 'True' \
                --early_stopping_patience 50 \
                --criterion 'cross-entropy' \
                --lr 0.00039 \
                --auto_lr 'False' \
                --log_dir 'finbenthic1' \
                --out_folder 'outputs' \
                --out_prefix 'finbenthic1-2-base-200-frz' \
                --deterministic 'True' \
                --ckpt_path "outputs/finbenthic1/finbenthic1-2-base-200_efficientnet_b0/f0/finbenthic1-2-base-200_efficientnet_b0_f0_230906-1647-133a_epoch153_val-loss0.17.ckpt" \
                --resume 'False'