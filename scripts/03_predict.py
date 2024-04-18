import argparse
import sys

import taxonomist as src


def main(args):
    parser = argparse.ArgumentParser()

    parser = src.add_dataset_args(parser)
    parser = src.add_dataloader_args(parser)
    parser = src.add_model_args(parser)
    parser = src.add_train_args(parser)
    parser = src.add_program_args(parser)

    args = parser.parse_args()

    user_arg_dict = vars(args)

    return src.TaxonomistModel(src.TaxonomistModelArguments(**user_arg_dict)).predict()


if __name__ == "__main__":
    trainer = main(sys.argv[1:])
