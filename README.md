# 🧠 Practical Deep Learning — Summer Semester 2026

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange?logo=jupyter)](https://jupyter.org/)

Welcome to the official code repository for the **Practical Deep Learning (pdl)** course, Summer Semester 2026.

This repo contains hands-on Jupyter notebooks, Python scripts, and utilities that accompany each lecture session. Every session builds on the previous one, guiding you from PyTorch fundamentals all the way to modern deep learning architectures.

---

## 📚 Course Structure

| # | Session | Topics |
|---|---------|--------|
| 01 | [Setup & Python Refresher](01_setup/) | Environment setup, NumPy refresher, intro to PyTorch tensors |
| 02 | [PyTorch Basics](02_pytorch_basics/) | Tensors, autograd, computation graphs, GPU usage |
| 03 | [Linear Models](03_linear_models/) | Linear & logistic regression, loss functions, gradient descent |
| 04 | [Neural Networks](04_neural_networks/) | MLPs, activation functions, BatchNorm, Dropout, weight init |
| 05 | [Convolutional Neural Networks](05_cnns/) | Conv layers, pooling, classic CNN architectures (LeNet, VGG, ResNet) |
| 06 | [Transfer Learning](06_transfer_learning/) | Fine-tuning pretrained models, feature extraction, torchvision |
| 07 | [Sequence Models](07_sequence_models/) | RNNs, LSTMs, GRUs, and an intro to the Transformer architecture |
| 08 | [Advanced Topics](08_advanced_topics/) | Generative models (VAE, GAN), model deployment, ONNX export |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or newer
- [conda](https://docs.conda.io/en/latest/miniconda.html) **or** `pip` (virtual environment recommended)
- A CUDA-capable GPU is optional but recommended for sessions 05–08

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate pdl_ss26
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Launch Jupyter

```bash
jupyter lab
```

Then open the notebook for the session you are working on from the file browser on the left.

---

## 🗂️ Repository Layout

```
pdl_ss26/
├── 01_setup/                  # Session 01 — Setup & Python Refresher
├── 02_pytorch_basics/         # Session 02 — PyTorch Basics
├── 03_linear_models/          # Session 03 — Linear Models
├── 04_neural_networks/        # Session 04 — Neural Networks
├── 05_cnns/                   # Session 05 — Convolutional Neural Networks
├── 06_transfer_learning/      # Session 06 — Transfer Learning
├── 07_sequence_models/        # Session 07 — Sequence Models
├── 08_advanced_topics/        # Session 08 — Advanced Topics
├── utils/                     # Shared helpers (plotting, data loading, …)
├── data/                      # Local datasets (ignored by git, see .gitignore)
├── environment.yml            # Conda environment spec
├── requirements.txt           # pip requirements
├── LICENSE
└── README.md
```

---

## 💡 How to Use the Notebooks

Each session folder contains:

- **`README.md`** — learning objectives and further reading links for the session
- **`*.ipynb`** — one or more Jupyter notebooks with explanations, code cells, and exercises
- **`solutions/`** *(where applicable)* — completed exercise solutions

Work through the notebooks top to bottom. Cells marked with `# TODO` are exercises for you to complete.

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` / `torchaudio` | Core deep learning framework |
| `numpy` | Numerical computing |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Classical ML utilities & metrics |
| `jupyter` / `jupyterlab` | Interactive notebooks |
| `tqdm` | Progress bars |
| `Pillow` | Image I/O |

See [`requirements.txt`](requirements.txt) or [`environment.yml`](environment.yml) for pinned versions.

---

## 🤝 Contributing

Found a bug or want to improve an exercise? Feel free to open an [issue](../../issues) or submit a pull request. Please follow the existing code style and add a short description of what you changed.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
© 2026 [LOMA-AImotion](https://github.com/LOMA-AImotion)
