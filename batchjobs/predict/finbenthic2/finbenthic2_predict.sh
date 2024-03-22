#!/bin/bash
#SBATCH --job-name=pred
#SBATCH --account=Project_2004353
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb0_pred.txt"
#SBATCH -e "e_eb0_pred.txt"

echo "Extracting data..."
unzip -q data/raw/FIN-Benthic2.zip -d $TMPDIR
echo "Done!"
source tykky
srun python scripts/03_predict.py \
    --data_folder "$TMPDIR/IDA/" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 1 \
    --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'geometric' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --ckpt_path "outputs/finbenthic2/finbenthic2-base-200_efficientnet_b0/f1/finbenthic2-base-200_efficientnet_b0_f1_230903-2143-d668_epoch07_val-loss0.18.ckpt"

## Test-time-augmentation
srun python scripts/03_predict.py \
    --data_folder "$TMPDIR/IDA/" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 1 \
    --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'geometric' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'True' \
    --out_prefix '' \
    --ckpt_path "outputs/finbenthic2/finbenthic2-base-200_efficientnet_b0/f1/finbenthic2-base-200_efficientnet_b0_f1_230903-2143-d668_epoch07_val-loss0.18.ckpt"