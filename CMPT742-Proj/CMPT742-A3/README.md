---
aliases: []
tags: []
icon:
iconColor:
date-created: 2025-11-17-08:46:20
date-modified: 2025-11-27-01:22:09
---

# CMPT 742 Assignment 3

This repository contains the SSD detector implementation used in Assignment 3. Use the bundled PowerShell helper `run.ps1` to dispatch the common workflows (training, evaluation, mAP calculation, and report-figure generation) without remembering the underlying Python entry points.

## Prerequisites

- Windows PowerShell (7.x recommended) with execution policy that allows local scripts.
- Python 3.10+ available on your `PATH` as `python` (the script forwards all work to this interpreter).
- Project dependencies installed in that interpreter: `pip install torch torchvision albumentations opencv-python numpy scikit-learn matplotlib` (match your CUDA build of PyTorch as needed).
- Dataset placed under `data/` following the existing `train/` and `test/` folder structure.

## Setup

1. Clone or download the repository and open a PowerShell prompt in the project root (`CMPT742-A3`).
2. (Optional but recommended) Create and activate a virtual environment.
3. Install Python dependencies in the active environment (see the `pip install …` command above or your preferred requirements file).
4. Verify that `python main.py --help` runs without module import errors before using the helper script.

## Using `run.ps1`

`run.ps1` exposes a single `-Mode` parameter with four choices. The script defaults to `train` when `-Mode` is omitted. Run it from the repository root so that relative paths to data and checkpoints resolve correctly.

| Mode    | Command       | Description |
| ------- | ------------- | ----------- |
| `train` | ```powershell |
./run.ps1 -Mode train
``` | Launches `main.py` for end-to-end training using the configuration hard-coded in that file. Checkpoints and logs are written to their respective folders (e.g., `checkpoints/`). |
| `test` | ```powershell
./run.ps1 -Mode test
``` | Runs `main.py --test`, loading the latest checkpoint and evaluating on the validation/test split configured in `dataset.py`. |
| `map` | ```powershell
./run.ps1 -Mode map
``` | Executes `gen_map.py` to compute mean Average Precision metrics and precision–recall curves. Ensure predicted boxes have been generated beforehand. |
| `gen_report` | ```powershell
./run.ps1 -Mode gen_report
``` | Calls `visualize.py` to refresh figures used in the final report (e.g., demo grids, confidence plots). |

### Common workflow

1. `./run.ps1 -Mode train` to train or fine-tune the model.
2. `./run.ps1 -Mode test` to verify qualitative/quantitative performance on the held-out images.
3. `./run.ps1 -Mode map` to collect mAP metrics for reporting.
4. `./run.ps1 -Mode gen_report` to regenerate the visuals before compiling `report/report.tex`.

### Troubleshooting tips

- If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` in the same session.
- Pass a fully qualified path to Python by editing `run.ps1` if `python` does not resolve to the environment you want.
- When experimenting on non-Windows hosts, mirror the commands by calling the underlying Python files directly (e.g., `python main.py --test`).

With the prerequisites installed, the script should provide a repeatable interface for the training and reporting flow expected in CMPT 742 Assignment 3.
