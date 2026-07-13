# MusicSage Bot

## Setup

- **Python**: 3.12, virtual env at `.venv/` (activate: `source .venv/bin/activate`)
- **Dep management**: no `pyproject.toml` or `requirements.txt` — install via `.venv` directly
- **External dep**: `ffmpeg` required on `$PATH` for audio extraction from `.stem.mp4` files
- **Dataset**: MUSDB18 at `musdb18/wav/{train,test}/` (gitignored). Each file is `.stem.mp4` with 5 audio streams (mix, drums, bass, other, vocals)

## Project structure

| Path | What |
|---|---|
| `model.py` | 2D U-Net definition (spectrogram in, 4 mask channels out, sigmoid) |
| `train.py` | Training script — uses `dataset.py` TFRecord pipeline |
| `inference.py` | CLI inference — STFT → predict masks → ISTFT with mix phase |
| `ver-1.ipynb` | Legacy 1D U-Net (waveform-in/waveform-out), **not compatible** with current pipeline |
| `data_preproccess/preprocess_data.py` | STFT preprocessing → `.npz` with masks, 10s chunks, FRAME_LENGTH=2048 |
| `data_preproccess/to_tf_record.py` | Raw waveform → sharded GZIP TFRecord writer |
| `data_preproccess/dataset.py` | TFRecord reader, on-the-fly STFT, masking, batching, normalization |
| `data_preproccess/compute_stats.py` | Compute dataset-wide mean/std for normalization |
| `musdb18/stft/` | Preprocessed `.npz` output (gitignored) |

## Key facts

- **Current model** (`model.py`): 2D U-Net, 3 encoder/decoder levels with 16-32-64-128-64-32-16 filters, `Conv2D(4, 1, sigmoid)` output, Adam(lr=1e-4), MSE loss
- **Legacy model** (`ver-1.ipynb` cell 25): 1D U-Net, waveform-in/waveform-out, **not compatible** with current pipeline
- **Training workflow**: (1) `to_tf_record.py` → (2) `compute_stats.py` → (3) `train.py`
- **Inference**: `inference.py input.mp3` — STFT → predict masks → ISTFT with mix phase
- **Multi-resolution STFT**: `MR_STFT_CONFIGS` in `dataset.py` and `inference.py` defines N `(frame_length, frame_step)` pairs. Reference (first config) is used for mask computation; all mix STFTs are resized to reference grid and stacked along channel dim → model input `(T, F, N)`. Add/remove entries to change channel count.
- **Preprocessing constants** differ between `preprocess_data.py` and the notebook — check before reusing
- **No tests, no CI/CD, no linting/formatting config**
- **Telegram bot + LLM (LangChain/LLaMA) not yet implemented** — only the ML pipeline and model exist
