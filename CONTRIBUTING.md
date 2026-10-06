# Contributing to Med-X

Thanks for your interest in improving Med-X, a multi-label chest X-ray classifier (DenseNet121 + Grad-CAM) trained on NIH ChestX-ray14. Bug reports, fixes, documentation and new experiments are all welcome.

Please keep one thing in mind: this is a research and learning project, **not a medical device**. Contributions must never present the model as clinically validated or suitable for diagnosis.

## Ways to contribute

- **Report a bug** — something crashes, gives wrong output, or the docs don't match the code.
- **Improve the model or evaluation** — ideas from the [roadmap](README.md#roadmap) are a good start: evaluation on the official NIH test list, better class-imbalance handling, probability calibration, external validation.
- **Improve the code** — tests, CI, cleaner structure, faster inference.
- **Improve the docs** — clearer setup steps, typo fixes, better explanations.

If you're unsure whether an idea fits, open an issue and ask before you start.

## Reporting bugs

Open an [issue](https://github.com/Harsh-Prajapati54/Medical_X/issues) and include:

- What you did, what you expected, and what happened instead
- The exact command or steps to reproduce it
- Your environment: OS, Python version, PyTorch version, CPU or GPU
- The full error message or traceback

**Do not upload real patient X-rays or anything that could identify a patient** to issues or pull requests. Use images from the public NIH dataset, or describe the problem in words.

## Suggesting changes

For anything bigger than a small fix (new model, new dataset, restructuring), please open an issue first so we can agree on the approach before you spend time on it.

## Development setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/Medical_X.git
   cd Medical_X
   ```
2. Create a virtual environment (Python 3.10+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
4. Run the app to check everything works:
   ```bash
   streamlit run app.py
   ```

The NIH ChestX-ray14 dataset is **not** in this repo. To train or evaluate, download it from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) or the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) and follow its terms of use.

## Making a pull request

1. Create a branch from `main` with a descriptive name: `fix/gradcam-device-error`, `feat/temperature-scaling`, `docs/setup-steps`.
2. Make your change. Keep each pull request focused on one thing.
3. Test it. At minimum, run the app and an inference on a sample image. If you add logic, add a small test (we use `pytest`; tests go in `tests/`).
4. Commit with short, clear messages in the imperative: `Fix Grad-CAM crash on CPU`, not `fixed stuff`.
5. Push to your fork and open a pull request against `main`. Explain what changed and why, and link the related issue.

### Pull request checklist

- [ ] The change is focused and the description explains the reason for it
- [ ] The app and inference still run
- [ ] No data, large files or secrets are committed (see below)
- [ ] Docs and the README are updated if behavior or results changed
- [ ] Notebook outputs are cleared, unless the outputs are the reference run

## Rules for ML changes

Changes to the model, data pipeline, or training need extra care so that results stay trustworthy.

- **Report numbers fully.** If a change affects performance, report per-class AUROC, AUPRC and F1 plus the macro averages, not just one metric or one favorable class. Accuracy alone is not meaningful here because the labels are heavily imbalanced.
- **Use the same evaluation protocol.** Compare against the current results on the same split. State the split, the random seed, and the hardware.
- **No test-set tuning.** Choose thresholds and hyperparameters on validation data only. Split by patient, not by image, so the same patient never appears in both train and validation.
- **Update results together with code.** If your change shifts the metrics, update `results/validation_metrics.csv` and the results tables in the README in the same pull request.
- **Be reproducible.** Set seeds, put settings in a config or at the top of the notebook, and note the library versions you used.
- **Don't cherry-pick.** If you report the best of several runs, say so and show how many runs you did.

## Files that should not be committed

- Datasets or any patient images
- Model checkpoints larger than a few tens of MB. Share them through GitHub Releases or the Hugging Face Hub and link to them instead (GitHub rejects files over 100 MB)
- API keys, tokens, or `.env` files (for example Weights & Biases keys)
- Generated outputs such as `gradcam_result.jpg`, logs, or `__pycache__`

## Code style

- Follow PEP 8 and keep functions small and readable.
- Add type hints and short docstrings to new functions.
- Prefer clear names over comments that explain unclear names.
- If you have Black or Ruff installed, run them on the files you touched.

## Review process

The maintainer reviews pull requests on a best-effort basis. You may be asked for changes or more evidence for a claimed improvement. Please be patient and respectful. Everyone here is volunteering their time.

## License

By contributing, you agree that your contributions will be licensed under the project's [Apache License 2.0](LICENSE).

## Questions

Open an [issue](https://github.com/Harsh-Prajapati54/Medical_X/issues) with your question. Thanks for helping make Med-X better!
