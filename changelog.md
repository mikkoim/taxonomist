# 0.0.3
### Documentation:
- Updated README
- Updates to workflows: RODI and Finbenthic2

### Preprocessing changes:
- Added random state option to `01_train_test_split.py`

### Train script changes:
- Explicit setting of `wandb.init()` so that it works in multi-GPU environments

### Prediction script changes:
- Feature extraction is now possible in `03_predict.py`. Setting `--feature_extraction` to "pooled" or "unpooled" returs a pickled file containing the feature outputs of the DNN, before a classification head.
- Feature extraction can be also done with pretrained models, if no checkpoint is passed to prediction script.
- Added possibility of returning logits instead of sigmoid probabilities, using `--return_logits 'True'` in prediction script

### Post-processing changes:
- Added an argument `--suffix` to CV prediction combination, so that specification between grouped and non-grouped predictions in the same folder can be distinguished.
- Added `--out_prefix` to evaluation script. `metrics` by default.

### Other changes
- Removed imsize as a parameter to `Dataset` and `LitDataModule`. This is passed via `aug_args`.
- Added segmentation module to the package. Documentation coming later.

### Environment:
- Added onnx, onnxruntime, biopython, networkx and pycocotools to environment

# 0.0.2

### Examples:
- Added FinBenthic1 examples

### Train script changes:
- Loading pretrained weights without resuming a previous run is now possible
- Last model checkpoint is saved
- Learning rate monitoring

### Prediction scripts changes:
- Fixed bug in TTA. Changes to DataModules were also made in `taxonomist.__init__`

### Post-processing changes:
- Fixed a bug in grouping script where reference dataset was not checked properly

### Evaluation changes:
- Running evaluation without bootstrap is now possible
- Re-designed and simplified comparison script