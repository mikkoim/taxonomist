import argparse

from taxonomist.input_config import add_compare_args
from taxonomist.evaluate import compare, CompareArgs

DESCRIPTION = """
Combines evaluation results for several models

Input:
    yaml file containing results to be compared in the structure. User input <inside brackets>:

    models:
        <model-1-name>:
            predictions: <path to model-1 predictions>
            metrics: <path to model-1 metrics>
            dataset: <The dataset that was used for testing (NOT training).
                        Used to differentiate different lenght outputs>
            length: <a length tag that differentiates different length outputs.
                     for example: "grouped" or "separate">
            tags:
                <tag-1>: <Tags are arbitary and can have arbitary values>
                <tag-2>: <tag 2 value>
        <model-2-name>:
            predictions:
            metrics:
            dataset:
            length:
                    
Output:
    Creates two sub-folders to out_folder: predictions and metrics. Predictions are grouped together
    by dataset-length pairs
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser = add_compare_args(parser)

    args = parser.parse_args()

    compare(CompareArgs(**vars(args)))
