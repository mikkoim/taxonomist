import argparse

from taxonomist import input_config, preprocessing, predictions, evaluate
import taxonomist 


def handle_train_test_split(args):
    arg_dict = vars(args)

    valid_keys = preprocessing.TrainTestSplitArgs.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = preprocessing.TrainTestSplitArgs(**filtered_dict)
    preprocessing.train_test_split(args)

def handle_train(args):
    arg_dict = vars(args)

    valid_keys = taxonomist.TaxonomistModelArguments.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = taxonomist.TaxonomistModelArguments(**filtered_dict)
    taxonomist.TaxonomistModel(args).train()

def handle_predict(args):
    arg_dict = vars(args)

    valid_keys = taxonomist.TaxonomistModelArguments.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = taxonomist.TaxonomistModelArguments(**filtered_dict)
    taxonomist.TaxonomistModel(args).predict()

def handle_combine_cv_predictions(args):
    arg_dict = vars(args)

    valid_keys = predictions.CombineCVPredictionsArgs.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = predictions.CombineCVPredictionsArgs(**filtered_dict)
    predictions.combine_cv_predictions(args)

def handle_group_predictions(args):
    arg_dict = vars(args)

    valid_keys = predictions.GroupPredictionsArgs.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = predictions.GroupPredictionsArgs(**filtered_dict)
    predictions.group_predictions(args)

def handle_evaluate(args):
    arg_dict = vars(args)

    valid_keys = evaluate.EvaluateArgs.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = evaluate.EvaluateArgs(**filtered_dict)
    evaluate.evaluate(args)

def handle_compare(args):
    arg_dict = vars(args)

    valid_keys = evaluate.CompareArgs.__annotations__.keys()
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_keys}

    args = evaluate.CompareArgs(**filtered_dict)
    evaluate.compare(args)

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(prog="taxonomist")

    subparsers = parser.add_subparsers(dest="command")

    # train_test_split arguments
    train_test_split_parser = subparsers.add_parser("train_test_split")
    train_test_split_parser = input_config.add_train_test_split_args(train_test_split_parser)
    train_test_split_parser.set_defaults(func=handle_train_test_split)

    # train arguments
    train_parser = subparsers.add_parser("train")
    train_parser = input_config.add_dataset_args(train_parser)
    train_parser = input_config.add_dataloader_args(train_parser)
    train_parser = input_config.add_model_args(train_parser)
    train_parser = input_config.add_train_args(train_parser)
    train_parser = input_config.add_program_args(train_parser)
    train_parser.set_defaults(func=handle_train)

    # predict arguments
    predict_parser = subparsers.add_parser("predict")
    predict_parser = input_config.add_dataset_args(predict_parser)
    predict_parser = input_config.add_dataloader_args(predict_parser)
    predict_parser = input_config.add_model_args(predict_parser)
    predict_parser = input_config.add_train_args(predict_parser)
    predict_parser = input_config.add_program_args(predict_parser)
    predict_parser.set_defaults(func=handle_predict)

    # combine_cv_predictions arguments
    combine_cv_predictions_parser = subparsers.add_parser("combine_cv_predictions")
    combine_cv_predictions_parser = input_config.add_combine_cv_predictions_args(combine_cv_predictions_parser)
    combine_cv_predictions_parser.set_defaults(func=handle_combine_cv_predictions)

    # group predictions arguments
    group_predictions_parser = subparsers.add_parser("group_predictions")
    group_predictions_parser = input_config.add_group_predictions_args(group_predictions_parser)
    group_predictions_parser.set_defaults(func=handle_group_predictions)

    # evaluate arguments
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser = input_config.add_evaluate_args(evaluate_parser)
    evaluate_parser.set_defaults(func=handle_evaluate)

    # compare arguments
    compare_parser = subparsers.add_parser("compare")
    compare_parser = input_config.add_compare_args(compare_parser)
    compare_parser.set_defaults(func=handle_compare)


    args = parser.parse_args()
    args.func(args)
