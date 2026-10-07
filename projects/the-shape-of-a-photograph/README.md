# The shape of a photograph

**Luiz Carazolli · Exploratory computational research**

Can the topology of an image's intensity landscape help distinguish real from AI-generated images? This project studies cubical persistent homology on CIFAKE and compares scalar topological summaries with a compact CNN operating on sampled persistence surfaces.

## Read first

- `report.pdf`: compiled public edition of the technical report.
- `report.tex` and `figures/`: complete, editable report source and figures.
- `PUBLICATION-NOTES.md`: corrections, provenance, and limits of the saved experiment.
- `the-shape-of-a-photograph.code-workspace`: portable editor workspace; open after extracting the full ZIP. It contains no absolute paths or automatic tasks.
- `figures/classification_metrics_benchmark_with_cnn.csv`: saved metrics for all seven classifiers.
- `results/topo_cnn_predictions.npz`: anonymized held-out labels and probabilities, allowing independent metric verification without the raw images or model files.
- `verify_results.py`: verifies saved predictions, key feature behavior, and published metrics.
- `requirements.txt`: dependency ranges (including the directly imported joblib package); `requirements-observed.txt`: exact versions observed in the supplied environment during release preparation.
- `CITATION.cff`: citation metadata.

## Main results

The input pool contains **20,000 images**, balanced between real and synthetic classes. The scripts re-split the downloaded CIFAKE **test partition** into 16,000 training-pool and 4,000 holdout images. The CNN uses 14,400 for fitting and 1,600 for validation within that training pool. This is an internal experiment, not evaluation using CIFAKE's official train/test protocol.

| Model | Representation | Holdout accuracy | ROC-AUC |
|---|---|---:|---:|
| RBF SVM | 16 scalar descriptors | 70.025% | 0.775485 |
| LightGBM | 16 scalar descriptors | 69.950% | 0.778880 |
| TopoCNN | Two 32 × 32 sampled surfaces | 74.700% | 0.8403695 |

The best scalar accuracy and best scalar AUC belong to different models. TopoCNN has 122,914 trainable parameters. The positive class is **real (1)**; synthetic is **AI (0)**. Saved TopoCNN recall for real images is 81.75%.

The code historically calls its tensors “persistence images.” It actually **samples** lifetime-weighted Gaussian surfaces on a fixed grid (bandwidth 0.05); it does not integrate them over cells. The introductory blog describes the standard integrated construction. Neither representation nor these results establish physical causation or a general-purpose detector.

## Setup

The supplied environment inspected for this edition uses Python 3.14.7. Exact numerical results may vary across dependency versions and hardware. Create a fresh environment in this extracted folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_results.py
```

For an environment matching the inspected package versions, use `requirements-observed.txt` instead. This is an inventory of the supplied environment, not proof of the versions used for the historical training run. Windows users can activate with `.venv\Scripts\activate`.

The pipeline's downloader requires `curl`. GUDHI, PyTorch, XGBoost, and LightGBM availability depends on your Python/platform combination. Report compilation requires a TeX distribution with `latexmk` and pdfLaTeX.

## Obtain the data and reproduce the pipeline

The scripts download the public Parquet test partition from the [CIFAKE mirror on Hugging Face](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset). Consult the [CIFAKE paper](https://arxiv.org/abs/2303.14126) and upstream terms for dataset provenance and reuse. Raw images, generated tensors, and trained checkpoints are intentionally not bundled.

Run from the project root, in this order:

```sh
# Downloads the image partition, then extracts 16 topology features + pixel summaries.
python pipeline.py --workers 4
python validate_dataset.py
python analyze_distributions.py

# Fits six scalar-feature baselines; saves train/test splits, models, metrics, figures.
python train_classifiers.py
python explain_topology_shap.py

# Builds two-channel sampled surfaces, fits TopoCNN, and redraws the combined ROC plot.
python generate_persistence_images.py --workers 4
python train_cnn_persistence.py
python plot_combined_roc.py

# Compile the report with cross-references resolved.
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

A full run can use substantial CPU and memory; GPU support is optional (MPS, CUDA, or CPU). The split seed is 42. The original CNN does not seed weight initialization or minibatch shuffling, so exact retraining metrics are not guaranteed. It runs 22 epochs and selects the checkpoint with the best validation AUC; it does not stop early.

For a quick extraction smoke test, use `python pipeline.py --sample-size 20 --workers 1 --output-dir data/sample`. Do not substitute a sampled output into the full benchmark and expect the published values.

**Re-running training overwrites local models, plots, and metric CSVs.** Preserve this extracted release or run experiments in a copy. The report contains the saved historical metrics; it does not automatically update its tables after retraining.

## What each script does

| File | Purpose |
|---|---|
| `pipeline.py` | Image ingestion, grayscale conversion, cubical persistence, scalar summaries |
| `validate_dataset.py` | Dataset integrity and exploratory statistics |
| `analyze_distributions.py` | Distribution plots and non-parametric tests |
| `train_classifiers.py` | Six tabular classifiers and five-fold training-pool cross-validation |
| `explain_topology_shap.py` | XGBoost feature attributions and interactions |
| `generate_persistence_images.py` | Persistence diagrams and grid-sampled Gaussian surfaces |
| `train_cnn_persistence.py` | Residual TopoCNN, validation checkpoint selection, evaluation |
| `plot_combined_roc.py` | Combined ROC comparison from fitted models and saved predictions |
| `verify_results.py` | Small offline numerical checks on the release |

## Evaluation limits

- Only one source dataset/generator setup and one saved CNN run are represented.
- No comparable raw-pixel CNN baseline, cross-generator evaluation, uncertainty intervals, or compression/blur robustness results are included.
- Exploratory distributions and SHAP analysis include the holdout; a fresh untouched evaluation is needed for confirmatory claims.
- SHAP describes predictions, not the physical causes of real/synthetic differences.
- Brier score measures overall probabilistic error; it is not a calibration-only measure.

## Attribution and reuse

Author: Luiz Otávio de Oliveira Carazolli. Use `CITATION.cff` for project attribution. Dependencies and upstream datasets retain their own terms. No additional software reuse license is declared in this release.
