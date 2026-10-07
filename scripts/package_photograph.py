#!/usr/bin/env python3
"""Prepare the downloadable, source-only release for The shape of a photograph."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "the-shape-of-a-photograph"
BUILD = ROOT / "tmp" / "project-publication" / "build" / "report.pdf"
ASSET = ROOT / "assets" / "projects" / "the-shape-of-a-photograph"

FIGURES = [p for p in (PROJECT / "figures").iterdir() if p.suffix.lower() in {".png", ".csv"}]
PUBLIC_FILES = [
    "README.md", "PUBLICATION-NOTES.md", ".gitignore", "CITATION.cff",
    "requirements.txt", "requirements-observed.txt", "verify_results.py",
    "the-shape-of-a-photograph.code-workspace", "report.tex",
    "pipeline.py", "validate_dataset.py", "analyze_distributions.py",
    "train_classifiers.py", "explain_topology_shap.py",
    "generate_persistence_images.py", "train_cnn_persistence.py",
    "plot_combined_roc.py",
]


def main():
    if not BUILD.exists():
        raise SystemExit(f"Compiled report missing: {BUILD}")
    ASSET.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD, PROJECT / "report.pdf")
    shutil.copy2(BUILD, ASSET / "report.pdf")
    for filename in ("README.md", "PUBLICATION-NOTES.md", "CITATION.cff"):
        shutil.copy2(PROJECT / filename, ASSET / filename)
    for name in ("classification_metrics_benchmark_with_cnn.csv", "persistence_images_samples.png", "roc_curves_with_cnn.png"):
        shutil.copy2(PROJECT / "figures" / name, ASSET / name)
    with zipfile.ZipFile(ASSET / "the-shape-of-a-photograph-workspace.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in PUBLIC_FILES:
            archive.write(PROJECT / name, name)
        for figure in sorted(FIGURES):
            archive.write(figure, f"figures/{figure.name}")
        archive.write(PROJECT / "results" / "verification.json", "results/verification.json")
        archive.write(PROJECT / "results" / "topo_cnn_predictions.npz", "results/topo_cnn_predictions.npz")
        archive.write(PROJECT / "report.pdf", "report.pdf")
    print(f"Packaged {ASSET / 'the-shape-of-a-photograph-workspace.zip'}")


if __name__ == "__main__":
    main()
