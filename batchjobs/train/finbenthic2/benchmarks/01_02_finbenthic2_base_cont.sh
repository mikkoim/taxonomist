#!/bin/bash
#SBATCH --job-name=base02
#SBATCH --account=Project_2004353
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_base_02.txt"
#SBATCH -e "e_base_02.txt"

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
                --aug 'trivialaugment' \
                --load_to_memory 'False' \
                --model 'efficientnet_b0' \
                --freeze_base 'False' \
                --pretrained 'True' \
                --opt 'adamw' \
                --max_epochs 100 \
                --min_epochs 5 \
                --early_stopping 'False' \
                --early_stopping_patience 50 \
                --criterion 'cross-entropy' \
                --lr 0.000229 \
                --auto_lr 'False' \
                --log_dir 'benchmarks' \
                --out_folder 'outputs' \
                --out_prefix 'finbenthic2-base-02' \
                --deterministic 'True' \
                --ckpt_path "outputs/finbenthic2/finbenthic2-base_efficientnet_b0/f0/finbenthic2-base_efficientnet_b0_f0_240514-0135-ef74_epoch52_val-loss0.28.ckpt"