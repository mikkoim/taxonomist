import argparse
import taxonomist as src
from joblib import Parallel, delayed
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def resize_image(imsize, fpath, folder, out_folder):
    img = Image.open(fpath)
    relpath = fpath.relative_to(folder)
    out_folder = out_folder / relpath.parents[0]
    out_folder.mkdir(exist_ok=True, parents=True)
    img.resize((imsize, imsize)).save(out_folder / relpath.name)

def main(args):
    dataset_config_module = src.utils.load_module_from_path(args.dataset_config_path)

    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=False, parents=True)

    fpaths = []
    for fold in range(args.n_folds):
        print(f"Reading fold {fold}")
        fpaths_fold, labels = dataset_config_module.preprocess_dataset(
            data_folder=args.data_folder,
            dataset_name=args.dataset_name,
            csv_path=args.csv_path,
            fold=fold,
            label=args.label_column,
        )
        fpaths.append(fpaths_fold["test"])
    
    fpaths = [x for xs in fpaths for x in xs]
    result = Parallel(n_jobs=-1)(delayed(resize_image)(args.imsize, fpath, Path(args.data_folder), out_folder) for fpath in tqdm(fpaths))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder")
    parser.add_argument("--dataset_config_path")
    parser.add_argument("--dataset_name")
    parser.add_argument("--csv_path")
    parser.add_argument("--label_column")
    parser.add_argument("--out_folder")
    parser.add_argument("--n_folds", type=int)
    parser.add_argument("--imsize", type=int)

    args = parser.parse_args()
    main(args)