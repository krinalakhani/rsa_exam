# Copilot Instructions

## Project shape
- This repository contains a single Jupyter notebook, `23CS038_final_exam.ipynb`, with no separate Python package or application entry point.
- Keep notebook cells in their top-to-bottom task order; later cells depend on variables created earlier.
- The notebook compares a non-private MNIST CNN with Opacus differential-privacy variants for a postal-department case study.

## Data and model flow
- Setup constants are `BATCH_SIZE = 256`, `EPOCHS = 3`, `DELTA = 1e-5`, `MAX_GRAD_NORM = 1.0`, and `DEVICE = torch.device('cpu')`.
- MNIST downloads to `./data`; training uses the first 20,000 examples and testing uses the full MNIST test set.
- Normalize with `(0.1307,)` and `(0.3081,)` using `torchvision.transforms.Compose([transforms.ToTensor(), ...])`.
- `PrivateCNN` uses two convolution blocks, `nn.GroupNorm(4, ...)`, ReLU, and max pooling. Preserve GroupNorm because Opacus DP training is incompatible with BatchNorm.
- Shared `train_one_epoch` and `evaluate_model` functions are used by baseline and DP training.

## Differential privacy patterns
- Target-budget runs use `PrivacyEngine.make_private_with_epsilon(...)` for epsilon targets `10`, `5`, and `1`.
- Fixed-noise runs use `make_private(...)` with `noise_multiplier=1.1`, recording `privacy_engine_fixed.get_epsilon(DELTA)` after each epoch.
- Keep `eps_spent_*` distinct from `test_acc_*` variables.
- Opacus-wrapped modules may expose the original model through `_module`; unwrap with `model._module if hasattr(model, '_module') else model`.
- Membership inference uses the mean maximum-softmax confidence gap between training and test loaders.

## Notebook workflow
- Restart the kernel and run every code cell sequentially; the notebook has no build system or `requirements.txt`.
- Install Opacus from the setup cell with `!pip install -q opacus`; first-run MNIST access requires network connectivity.
- Run the notebook relative to the notebook directory. Generated files are `accuracy_vs_epsilon.png` and `epsilon_vs_epoch.png`.

## Editing and validation
- Do not introduce BatchNorm into `PrivateCNN` for the Opacus workflow.
- `PrivateCNN` is currently duplicated across task cells; update both definitions or consolidate to one canonical definition.
- Preserve `acc_cost` before it is referenced in Task 6: `acc_cost = test_acc_1 - baseline_acc`.
- Preserve result labels and plot filenames used for the empirical report.
- Validate with a full top-to-bottom execution and confirm baseline/DP accuracy, epsilon accounting, MIA values, and both PNG outputs.
