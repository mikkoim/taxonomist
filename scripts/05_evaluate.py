import argparse

from taxonomist.input_config import add_evaluate_args
from taxonomist.evaluate import evaluate, EvaluateArgs

DESCRIPTION = """
Calculates metrics to prediction outputs.

Input:
A csv file containing true and predicted labels
A config file containing metrics that are calculated

Output:
A dataframe containing 
    metrics
    bootstrapped confidence intervals
for
    full cv predictions
    each fold separately
"""




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser = add_evaluate_args(parser)

    args = parser.parse_args()

    evaluate(EvaluateArgs(**vars(args)))

