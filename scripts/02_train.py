import argparse
import sys

import taxonomist as src

DESCRIPTION = """
The main training script.
"""


def main(args):
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser = src.add_dataset_args(parser)
    parser = src.add_dataloader_args(parser)
    parser = src.add_model_args(parser)
    parser = src.add_train_args(parser)
    parser = src.add_program_args(parser)

    args = parser.parse_args(args)

    user_arg_dict = vars(args)
    return src.TaxonomistModel(
        src.TaxonomistModelArguments(**user_arg_dict)
    ).train_model()


if __name__ == "__main__":
    trainer = main(
        sys.argv[1:]
    )  # returns pytorch lightning trainer that has been trained.
