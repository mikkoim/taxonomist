import argparse

from taxonomist.input_config import add_group_predictions_args
from taxonomist.predictions import group_predictions, GroupPredictionsArgs

DESCRIPTION = """
Performs aggregation to predictions, based on a group variable in the original dataset.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser = add_group_predictions_args(parser)

    args = parser.parse_args()

    group_predictions(GroupPredictionsArgs(**vars(args)))

