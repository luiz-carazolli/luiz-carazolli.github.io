# Public edition notes · 6 October 2026

The public workspace was prepared from the existing local project. Original files were preserved. `main.tex` and `report.tex` were identical; `report.tex` is the canonical source in this release. The old duplicate sources, prebuilt archive, virtual environment, caches, raw datasets, and trained model checkpoints are omitted.

## Report corrections

- Added the author's name and a fixed public-edition date.
- Made the evaluation protocol explicit: an internal re-split of the upstream 20,000-image **test** partition, not the official benchmark protocol.
- Corrected the representation description: the actual tensors are sampled Gaussian persistence surfaces, not cell-integrated persistence images.
- Preserved saved numerical results and figures; separated the best tabular accuracy (RBF SVM) from the best tabular AUC (LightGBM).
- Replaced claims of mathematical proof, physical causation, universal detection robustness, and calibration improvement with statements supported by the experiment.
- Corrected persistence entropy interpretation and the distinction between SHAP attribution shares and predictive accuracy.
- Clarified that all 22 CNN epochs run, with best-validation-AUC checkpoint selection, rather than early stopping.
- Documented unseeded CNN initialization/shuffling, exploratory use of holdout data, and missing out-of-distribution/robustness baselines.
- Adjusted report layout where necessary to prevent overflow.

## Code changes

Added the missing `Tuple` import in `train_cnn_persistence.py` for Python versions that evaluate annotations eagerly. The model, feature extraction, split logic, and training procedure are otherwise preserved. Added joblib explicitly to the dependency ranges. No new training was performed for this release.

## Checks

The release includes offline checks for feature behavior and saved CNN metrics. Preparation also compared sample raw-image features and surface tensors with their saved rows, checked dataset balance, and compared the train/test indices across the tabular and CNN representations. See `results/verification.json` for the actual checks completed.

The full report is compiled from the included source and figures. Historical figure titles using “Persistence Images” should be read in light of the sampled-surface clarification above.
