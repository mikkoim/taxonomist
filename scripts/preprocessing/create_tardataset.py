import os
import tarfile
from pathlib import Path
import argparse


def create_tar_files(args):
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"The path {args.folder} is not a valid directory.")
        return

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(list(folder.iterdir()))

    if args.shuffle:
        import random
        random.seed(0)
        random.shuffle(files)

    tar_index = 0
    file_count = 0
    tar_files = []
    tar_name = output_folder / f"{args.output_prefix}-{tar_index:05d}.tar"
    tar = tarfile.open(tar_name, "w")
    total_file_count = 0

    for file in files:
        if file.is_file():
            # Check if the current tar has reached the max number of files
            if file_count >= args.max_files:
                tar.close()  # Close the current tar file
                print(f"Created: {tar_name}")
                tar_files.append(tar_name)

                # Start a new tar file
                tar_index += 1
                tar_name = output_folder / f"{args.output_prefix}-{tar_index:05d}.tar"
                tar = tarfile.open(tar_name, "w")

                # reset the file count
                total_file_count += file_count
                file_count = 0

            # Add the file to the tar
            tar.add(file, arcname=file.name)
            file_count += 1

    total_file_count += file_count
    # Close the final tar if it contains any files
    if file_count > 0:
        tar.close()
        print(f"Created: {tar_name}")
        tar_files.append(tar_name)

    print(f"{len(tar_files)} tar files created")
    print(f"Total files: {total_file_count}")
    print(f"Last tar file has {file_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--max_files", type=int, default=10000)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    create_tar_files(args)