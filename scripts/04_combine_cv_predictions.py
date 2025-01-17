import argparse

from taxonomist.input_config import add_combine_cv_predictions_args
from taxonomist.predictions import combine_cv_predictions, CombineCVPredictionsArgs

DESCRIPTION = """
Combines the outputs from multiple cross-validation folds into a since prediction vector.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser = add_combine_cv_predictions_args(parser)

    args = parser.parse_args()

    combine_cv_predictions(CombineCVPredictionsArgs(**vars(args)))