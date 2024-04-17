This workflow describes general image classification for any data set 

## 1. Data loading

The general idea is to have all data in `data/raw/{your_data_set_identifier}`. There are examples for several datasets in `docs/dataset_loading.md`

## 2. Raw data analysis

The general idea is to have a separate scrip/notebook that shows how to examine the data. This step is not necessary, but recommended.
There are examples for examining the data in `notebooks/01_raw_data_analysis_{rodi OR finbenthic1 OR finbenthic2}.ipynb`

## 3. Preprocess the data
The purpose of this step is to generate a file that indicates how to find the needed information from the raw data file in `data/raw/{your_data_set_identifier}`. This happens by generating a file and saving it to `data/processed/{your_data_set_identifier}/{processed_data_file_name}.csv`. This file should be a `.csv`-file with at least three columns that indicate how to find the images from the raw data folder ({path_indicator}), what are the labels of those images ({label_indicator}), and what object is being imaged if many images are taken of the same object (`{group_indicator}`). It doesn't matter how these information are presented, since you will generate in later phases custom methods that use these infromation in the way you have inteded.

There can be additional columns and there is flexibility in how the wanted information is presented. For instance for the rodi-dataset, there are 7 columns and the column that indicates how to find the image (column name: image) is a string containing the name of the image (not the whole path). The column indicating the label (column name: family) is a string indicating the name of the class. The idea is to have all the data unmodified in the `data/raw/{your_data_set_identifier}` and to just point out where the needed data are. For rodi-data, the resulting file is 429KB whereas the original files are almost 500MB.

## 4. Train-test-splits
The purpose of this step is to divide the data to train, test and validation sets. There is an existing implementation in `scripts/01_train_test_split.py` and you can see its documentation for further information. The script requires you to give the path to the `.csv` file containing the information about the preprocessed data (generated in the previous step). It also requires the name of the column that show the labels and group indicator indicating which object is being imaged. In addition to these, the implementation asks for path where the resulting ´.csv´ of this step should be saved (e.g. `data/processed/{your_data_set_identifier}`) and what should be the name of the column where the splits are saved ({split_indicator}). The script always has the images of same object are in the same split as otherwise there would be too much correlation between the data in different splits. 

You can run the script as follows
```powershell
python scripts/01_train_test_split.py `
    --csv_path "data\\processed\\{your_data_set_identifier}\\{processed_data_file_name}.csv" `
    --target_col "{label_indicator}" `
    --group_col "{group_indicator}" `
    --out_folder "data\\processed\\{your_data_set_identifier}"
```

```bash
python scripts/01_train_test_split.py \
    --csv_path "data\\processed\\{your_data_set_identifier}\\{processed_data_file_name}.csv" \
    --target_col "{label_indicator}" \
    --group_col "{split_indicator}" \
    --out_folder "data/processed/{your_data_set_identifier}"
```

## 5. Training
The purpose of this step is to train the model using the data indicated in the previous step. 

```powershell
python -m pdb scripts\\02_train.py `
    --data_folder "data\\raw\\rodi\\Induced_Organism_Drift_2022" `
    --dataset_name "rodi" `
    --csv_path "data\\processed\\rodi\\01_rodi_processed_5splits_family.csv" `
    --label "family" `
    --fold 0 `
    --class_map "data\\processed\\rodi\\rodi_label_map.txt" `
    --imsize 224 `
    --batch_size 256 `
    --aug 'aug-02' `
    --load_to_memory 'False' `
    --model 'resnet18' `
    --opt 'adamw' `
    --max_epochs 200 `
    --min_epochs 5 `
    --early_stopping 'True' `
    --early_stopping_patience 10 `
    --criterion 'cross-entropy' `
    --lr 0.0001 `
    --auto_lr 'True' `
    --log_dir 'rodi' `
    --out_folder 'outputs' `
    --out_prefix 'rodi' `
    --deterministic 'True'
```

python scripts/02_train.py \
    --data_folder "data/raw/rodi/Induced_Organism_Drift_2022" \
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
    --lr 0.0001 \
    --auto_lr 'True' \
    --log_dir 'rodi' \
    --out_folder 'outputs' \
    --out_prefix 'rodi' \
    --deterministic 'True'