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