# MusicSage Bot

## Setup

- **Python**: 3.12, virtual env at `.venv/` (activate: `source .venv/bin/activate`)
- **Dep management**: no `pyproject.toml` or `requirements.txt` — install via `.venv` directly
- **External dep**: `ffmpeg` required on `$PATH` for audio extraction from `.stem.mp4` files (stempeg)
- **Dataset**: MUSDB18 at `musdb18/wav/{train,test}/` (gitignored). Each file is `.stem.mp4` with 5 audio streams (mix, drums, bass, other, vocals). Note: currently not downloaded on the dev machine — notebooks that need it must run on a machine with the dataset (or switch `stft_visualizer.ipynb` to `DATA_SOURCE="file"`)

## Project structure

### PyTorch pipeline (current)

| Path | What |
|---|---|
| `model.py` | `HybridUNet` — 1D WaveUNet (waveform, 36M params) + 2D STFTBranch (spectral masks, 2.6M), гейтованное объединение. Вход `(B, 2, N)` микс → выход `(B, 4, 2, N)` стемы |
| `separator_pytorch.ipynb` | **Основной трейнер** HybridUNet: musdb-данные с LRU-кэшем, Demucs-аугментации, loss (L1 + MR-spectral + SI-SDR + mixture consistency), свип 7 конфигов, инференс чанками с кросфейдом, SI-SDR метрика |
| `separator_21D_pytorch.ipynb` | Тот же пайплайн, но реверсивная архитектура: **2D первая** (маски → ISTFT с фазой микса) → **1D WaveUNetRefine вторая** (вход 8 каналов, residual на входные стемы) — каскад, не гейт |
| `stft_visualizer.ipynb` | Визуализация STFT предсказания модели vs идеальные стемы: настраиваемые `START_SEC`/`DISPLAY_SEC`, `VIS_NFFT`/`VIS_HOP`/`VIS_WINDOW`, `DATA_SOURCE` (musdb или свой файл + опц. `REF_DIR`) |
| `checkpoints_pytorch/sep_best.pt` | Обученный HybridUNet (64/5, SPEC 4096/1024). State_dict с префиксом `module.` (DataParallel) — **снимать при загрузке** |
| `checkpoints_pytorch/sweep/`, `checkpoints_pytorch/sweep_21d/` | Чекпойнты свипов (перезаписываются при запуске свипа) |

### TF legacy (несовместимо с PyTorch)

| Path | What |
|---|---|
| `inference.py` | CLI инференс TF-модели: STFT → предсказание масок → ISTFT с фазой микса → опц. 1D refine. `MR_STFT_CONFIGS` — мультирезолюционный STFT-стек `(T, F, N)` |
| `refine.py`, `train_refine.py` | 1D U-Net рефайнер (TF, mini Demucs) для waveform-чистки после ISTFT |
| `losses.py` | TF лоссы (`ComplexSeparatorLoss`) |
| `data_preproccess/` | TFRecord-пайплайн: `dataset.py` (ридер + on-the-fly STFT), `preprocess_data.py` (STFT → `.npz`), `to_tf_record.py` (шардированные TFRecord), `compute_stats.py` (mean/std), `preprocess.ipynb` |
| `ver-1.ipynb` | Legacy 1D U-Net waveform-in/waveform-out, **не совместим** с текущим пайплайном |
| `separator.ipynb` | Legacy TF-ноутбук обучения |
| `datasert_test.ipynb` | TF-датасет тест |
| `train.py` | Удалён — обучения через скрипт больше нет, только ноутбуки |

## Key facts

- **Текущая модель** (`model.py`): `HybridUNet` — 1D WaveUNet (Demucs v2 style, stride-4 + GLU, dilated bottleneck, глобальный residual `(1+dummies)·mix`) + 2D `STFTBranch` (2D U-Net + BiLSTM bottleneck по log-магнитуде STFT 4096/1024 → softmax-маски по стемам). Выход = `g·wave + (1−g)·spec`, обучаемый гейт; после обучения `sigmoid(gate) ≈ 0.33` — спектральная ветка доминирует. Итого ~38.6M (wave 36M, spec 2.6M)
- **Реверсивная модель** (`separator_21D_pytorch.ipynb`): `ReversedHybridUNet` — 2D маски → ISTFT (фаза микса) → 1D `WaveUNetRefine` (вход `(B, 4, 2, N)` = 8 каналов, residual на входные стемы). Аналог `refine.py`, но обучен end-to-end
- **Модельный интерфейс** (обе архитектуры): вход `(B, 2, N)` нормализованный микс, выход `(B, 4, 2, N)` стемы; длина паддится до кратного `4**LEVELS` (1024 для levels=5)
- **Инференс**: чанки 6 с, 50% overlap, Hann-кросфейд, нормализация per-chunk по std микса (как в трейнинге), опц. Wiener soft-mask пост-процессинг
- **Важно — AMP**: `run_epoch` использует `enabled=scaler is not None and USE_AMP`. `USE_AMP` только на CUDA; на MPS/CPU autocast нельзя включать (fp16 на входах BiLSTM → краш)
- **Аугментации** (Demucs-стиль): случайный гейн, FIR-эквалайзер, питч-шифт ±2.5 полутона, своп L/R, mixup стемов между треками (chunk swap)
- **Данные**: `musdb.DB` читает `*.stem.mp4` через stempeg, датасет блочный (чанки подряд из одной песни + LRU-кэш, `DataLoader(shuffle=False)`)
- **Проблема MPS**: `num_workers=2` + spawn-запуск скриптом зависает (воркеры перезапускают main) — в Jupyter ок; `pin_memory=True` на MPS даёт warning (безвредно)
- **Preprocessing constants** в TF-пайплайне отличаются от ноутбуков — не смешивать
- **Нет тестов, CI/CD, линтера**
- **Telegram bot + LLM (LangChain/LLaMA) не реализованы** — только ML-пайплайн и модели