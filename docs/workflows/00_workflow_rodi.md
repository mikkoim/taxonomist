This workflow describes image classification for the RODI dataset (de Schaetzen et al. 2023). With a GPU, the full workflow can be run in under 30 minutes, including training and prediction for multiple models.

The commands are written for unix command lines. To be able to copy-paste them in Windows, you have to change the backslashes: `\` into backtics: `´`, and change the forward slashes `/` in filenames into two backslashes: `\\`. Also, the export commands wont work so manual pasting of filenames in some scripts is needed.

## 1. Data loading

Load data like described in `docs/dataset_loading.md`

## 2. Raw data analysis

Analyze the raw dataset like in `notebooks/01_raw_data_analysis_rodi.ipynb`

## 3. Preprocess the data

The preprocessing script produces a file in `data/processed/rodi/01_rodi_processed.csv`.

```bash
export TMPDIR = "data/raw/rodi/" # on CSC this should be the normal nvme TMPDIR
python scripts/preprocessing/process_rodi.py \
    --csv_path="$TMPDIR/Induced_Organism_Drift_2022_annotations.csv" \
    --out_folder="data/processed/rodi"
```

## 4. Train-test-splits

The train-test-split script adds additional columns to the end of the dataset table.
These columns contain information about whether the image belongs to the test, train 
or validation set.

bash (Unix)
```bash
python scripts/01_train_test_split.py \
    --csv_path "data/processed/rodi/01_rodi_processed.csv" \
    --target_col "family" \
    --group_col "ind_id" \
    --n_splits 5 \
    --out_folder "data/processed/rodi"
```

## 5. Training
Running this training script should take 5-10 minutes with a GPU, running for around 20 epochs before stopping.
```bash
export TMPDIR = "data/raw/rodi/"
python scripts/02_train.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
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
    --lr 0.001 \
    --auto_lr 'True' \
    --log_dir 'roditest_new' \
    --out_folder 'outputs' \
    --out_prefix 'rodi_new' \
    --deterministic 'True'
```
It saves a single model trained with only the first fold to `outputs/rodi/rodi_new_resnet18/f0`
This folder then contains a config `config_<run_id>.yml` and two checkpoint files: last and the checkpoint with the lowest validation loss.
Visualizations of applied augmentations are seen in `aug-<augname>-<run_id>` folder.

### Training all folds
Easiest way to train all folds is to just define a simple bash script that loops through all folds:

Often you might first want to test training with a single fold, but for accurate results you will want to run the final model trainining with cross-validation.

This training should take around 10-20 minutes with a decent GPU.
```bash
for i in {0..4}
do
python scripts/02_train.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold $i \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'aug-02' \
    --load_to_memory 'False' \
    --tta 'False' \
    --model 'resnet18' \
    --opt 'adamw' \
    --max_epochs 5 \
    --min_epochs 0 \
    --early_stopping 'False' \
    --early_stopping_patience 0 \
    --criterion 'cross-entropy' \
    --lr 0.004 \
    --auto_lr 'False' \
    --log_dir 'roditest' \
    --out_folder 'outputs' \
    --out_prefix 'rodi-allfolds' \
    --deterministic 'True'
done
```
These outputs are saved to `outputs/rodi/rodi-allfolds_resnet18`, into 5 fold folders: f0, f1, f2, f3 and f4

## 6. Prediction
Next step is predicting test set values using the models trained in previous steps.

Note that now out_prefix does not have anything. This could be specified if we have different kinds of predictions.

Be especially careful that you set the fold to the correct one that you use your checkpoint with so that you don't test with same data the model was trained on.

The prediction produces an output csv that has the softmax probability for all of the classes in the dataset.
Thus, the output can get very large with large datasets with a lot of classes.
Prediction can also produce logit outputs with the parameter `--return_logits 'True'`

```bash
# This just sets the checkpoint as the first (best) model in the directory above, as the unique identifier is always different.
export CKPT_PATH=$(find "outputs/rodi/rodi_new_resnet18/f0/" -type f -name "rodi_new_resnet18*.ckpt" ! -name "*_last.ckpt" | head -1)
python scripts/03_predict.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --ckpt_path $CKPT_PATH
```

The predictions are saved to `outputs/rodi/rodi_new_resnet18/f0/predictions/rodi_none`
The output folder is named, based on the augmentation used, this case 'none'. If we want to use test-time-augmentation, the choice of augmentation has an effect on the results:

```bash
python scripts/03_predict.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'geometric' \
    --out_folder 'outputs' \
    --tta 'True' \
    --out_prefix '' \
    --ckpt_path $CKPT_PATH
```
This time test-time-augmentation is applied. Every image is classified 5 times, with a random augmentation applied.
Note that the parameters 'tta' and 'aug' are now different.
Outputs are saved to `rodi_geometric_tta`

Predictions are always saved to the fold folder of `dataset_name/` 

### Feature extraction

```bash
python scripts/03_predict.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold 0 \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix 'features' \
    --ckpt_path $CKPT_PATH \
    --feature_extraction "pooled"
```

The feature outputs are saved as a pickle file.

## 7. Combining cross-validation predictions

Let's perform prediction for all the 5 folds that we trained before.

The export commands are just to that this tutorial works by copy-pasting. It is also possible and probably more clear to just specify the checkpoint path manually

```bash
for i in {0..4}
do
export CKPT_PATH=$(find "outputs/rodi/rodi-allfolds_resnet18/f$i/" -type f -name "rodi-allfolds_resnet18_f$i_*.ckpt" ! -name "*_last.ckpt" | head -1)
echo $CKPT_PATH >> ckpts_used.txt
python scripts/03_predict.py \
    --data_folder "$TMPDIR/Induced_Organism_Drift_2022" \
    --dataset_name "rodi" \
    --csv_path "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --label "family" \
    --fold $i \
    --class_map "data/processed/rodi/rodi_label_map.txt" \
    --imsize 224 \
    --batch_size 1024 \
    --aug 'none' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix '' \
    --ckpt_path $CKPT_PATH
done
```

Now we should have prediction files in all five fold folders. The predictions are worse since we trained each model for only ten epochs and didn't fine-tune the learning rate.

Since the train-test-split function splits the dataset into test-folds that together make up the entire dataset, we can combine all the predictions from all folds to produce a prediction set for the entire dataset.

```bash
python scripts/04_combine_cv_predictions.py \
    --model_folder "outputs/rodi/rodi-allfolds_resnet18" \
    --reference_csv "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --reference_target "family" \
    --tag "rodi_none" \
    --n_folds 5
```

This script creates a function `outputs/rodi/rodi-allfolds_resnet18/prediction`, which contains the prediction file for the full dataset.

## 8. Grouping the predictions

Once we have a prediction file, either for a single fold of the full dataset, we can group the predictions if it has a grouping variable. This can be for example an individual that we have several images of.

Grouping for the single fold
```bash
export CKPT_PATH=$(find "outputs/rodi/rodi_new_resnet18/f0/" -type f -name "rodi_new_resnet18_f0_*.ckpt" ! -name "*_last.ckpt" | head -1)
export CKPT_STEM=$(basename "$CKPT_PATH" | sed 's/\.[^.]*$//')
python scripts/04_group_predictions.py \
    --predictions "outputs/rodi/rodi_new_resnet18/f0/predictions/rodi_none/_${CKPT_STEM}_none.csv" \
    --reference_csv "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --reference_target "family" \
    --fold 0 \
    --reference_group "ind_id" \
    --agg_func "mode"
```
outputs are saved to the same folder as predictions.

Grouping for the full dataset
```bash
python scripts/04_group_predictions.py \
    --predictions "outputs/rodi/rodi-allfolds_resnet18/predictions/rodi-allfolds_resnet18_rodi_none.csv" \
    --reference_csv "data/processed/rodi/01_rodi_processed_5splits_family.csv" \
    --reference_target "family" \
    --reference_group "ind_id" \
    --agg_func "mode"
```


## 9. Evaluation

Evaluation script uses a separate yaml file that specifies the metrics that are calculated. If additional metrics are desired, they can be specified as functions in the 05_evaluate.py -function.

Let's evaluate both the full dataset prediction with all images separately,
and grouped by individual.
```bash
# separate
python scripts/05_evaluate.py \
    --predictions "outputs/rodi/rodi-allfolds_resnet18/predictions/rodi-allfolds_resnet18_rodi_none.csv" \
    --metric_config conf/eval.yaml

# grouped
python scripts/05_evaluate.py \
    --predictions "outputs/rodi/rodi-allfolds_resnet18/predictions/rodi-allfolds_resnet18_rodi_none_grouped.csv" \
    --metric_config conf/eval.yaml
```

Outputs are saved to a new folder, `metrics` which is sibling of the `prediction`-folder the prediction csv is in.

Values are by default calculated with bootstrapped 95% confidence intervals. 
This means that the predictions are resampled to their full size, with replacement, and metric values are calculated for the resampled dataset.
If this is not desired, a parameter `--no_bootstrap` can be passed

Let's calculate metrics for the single fold predictions also so we have two 
outputs in different locations to compare.

```bash
python scripts/05_evaluate.py \
    --predictions "outputs/rodi/rodi_new_resnet18/f0/predictions/rodi_none/_${CKPT_STEM}_none_grouped.csv" \
    --metric_config conf/eval.yaml
```

## 10. Comparison

taxonomist makes it easy to compare metrics in different experiments.
Sometimes results are ran with a model and dataset combination, maybe with a specific grouping or a target label. The comparison script makes it possible to re-use metrics in different comparison scenarios, for example if it is wanted to name models differently for tables or figures.

```bash
python scripts/06_compare.py \
        --config 'conf/experiments/rodi_test_experiment.yaml' \
        --out_folder 'results'
```

compare.py pools together all the metrics files specified in the config file, and
saves these into `out_folder/metrics`. This file can then easily be analyzed in any 
software and shared easily.

The script also saves predictions for convenient comparison.
Different datasets have different lenghts, and each dataset can have a grouping, resulting again in a different length. The comparison script saves all dataset-length combinations as their own csv-files