#!/bin/bash
#SBATCH --job-name=2gpu
#SBATCH --account=Project_2004353
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:2,nvme:64
#SBATCH -o "o_2gpu.txt"
#SBATCH -e "e_2gpu.txt"

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
                --early_stopping_patience 0 \
                --criterion 'cross-entropy' \
                --lr 0.002754228703338169 \
                --auto_lr 'False' \
                --precision 16 \
                --log_dir 'benchmarks' \
                --out_folder 'outputs' \
                --out_prefix 'finbenthic2-2gpu' \
                --deterministic 'True' \
                --strategy "ddp"