# MambaVoiceCloning (MVC) - Efficient and Expressive TTS with State-Space Modeling

This paper presents **MambaVoiceCloning (MVC)**, a scalable and expressive text-to-speech (TTS) framework that unifies *state-space sequence modeling* with *diffusion-driven style control*. Distinct from prior diffusion-based models, MVC replaces all self-attention and recurrent components in the TTS pipeline with novel Mamba-based modules: a Bi-Mamba Text Encoder, Temporal Bi-Mamba Encoder, and Expressive Mamba Predictor. These modules enable linear-time modeling of long-range phonetic and prosodic dependencies, improving efficiency and expressiveness without relying on external reference encoders. While MVC uses a diffusion-based decoder for waveform generation, our contribution is architectural—introducing the first end-to-end Mamba-integrated TTS backbone. Extensive experiments on LJSpeech and LibriTTS demonstrate that MVC significantly improves naturalness, prosody, intelligibility, and latency over state-of-the-art methods. MVC maintains a lightweight footprint of 21M parameters and achieves 1.6× faster training than comparable Transformer-based baselines.

### 🎧 Audio Demos
Explore MVC's expressive and high-quality speech synthesis through our audio samples: [MVC Audio Demos](https://aiai-9.github.io/mvc1.github.io/)


## 🚀 Key Features

* **Attention-Free Architecture:** Fully eliminates attention and recurrence for linear-time efficiency.
* **State-Space Modeling:** Leverages Mamba-based SSMs for efficient long-range sequence modeling.
* **Expressive Speech Generation:** Captures fine-grained prosodic and speaker variations without external reference encoders.
* **Real-Time Inference:** Achieves sub-20ms latency for scalable, real-time TTS applications.
* **Multilingual and Multispeaker Support:** Designed to extend to multilingual and cross-lingual TTS with flexible style control.
* **Open Source and Reproducible:** Full code, pretrained models, and training scripts provided for transparency and ease of use.

## 📂 Repository Structure

```
MVC/
├── configs/                # Configuration files for training and evaluation
├── models/                 # Core model architectures (Bi-Mamba, Expressive Encoder)
├── data/                   # Dataset preparation and processing scripts
├── utils/                  # Utility functions for data loading, augmentation, and logging
├── experiments/            # Training scripts and experimental setups
├── results/                # Generated speech samples and evaluation logs
└── README.md               # Project overview and setup instructions
```

## 📦 Installation

### Prerequisites

* Python >= 3.8
* PyTorch >= 1.12.0
* CUDA-enabled GPU (recommended for training)

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/aiai-9/MVC.git
cd MVC
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the LJSpeech dataset [here](https://keithito.com/LJ-Speech-Dataset/) or the LibriTTS dataset [here](https://www.openslr.org/60/).
<!-- 2. Preprocess the data:

```bash
python scripts/preprocess.py --dataset LJSpeech --output_path data/LJSpeech
``` -->

### Training

First stage training (Text Encoder, Duration Encoder, Prosody Predictor):

```bash
accelerate launch train_first.py --config_path ./configs/config.yml
```

Second stage training (Diffusion-based decoder and adversarial refinement):

```bash
python train_second.py --config_path ./configs/config.yml
```

### Inference

Generate high-quality speech with pre-trained models:

```bash
python inference.py --config_path ./configs/config.yml --input_text "Hello, this is MambaVoiceCloning."
```

## 🧠 Model Architecture

MVC consists of three core components:

1. **Bi-Mamba Text Encoder:** Efficiently captures phoneme-level context using bidirectional state-space models (SSMs).
2. **Expressive Mamba Encoder:** Enhances prosodic variation and speaker expressiveness.
3. **Temporal Bi-Mamba Encoder:** Models rhythmic structures and duration alignment for natural speech generation.

![MVC Architecture](figures/mvc.png)

## 📊 Evaluation

Run objective and subjective evaluations using provided scripts:

```bash
python evaluate.py --config_path ./configs/config.yml
```

## 🏆 Results

### Table 1: Subjective Evaluation on LibriTTS (Zero-Shot)

| Model          | MOS-N (↑)       | MOS-S (↑)       |
| -------------- | --------------- | --------------- |
| Ground Truth   | 4.60 ± 0.09     | 4.35 ± 0.10     |
| VITS           | 3.69 ± 0.12     | 3.54 ± 0.13     |
| StyleTTS2      | 4.15 ± 0.11     | 4.03 ± 0.11     |
| **MVC (Ours)** | **4.22 ± 0.10** | **4.07 ± 0.10** |

### Table 2: MOS Comparison on LJSpeech (ID vs OOD)

| Model          | MOS\_ID (↑)     | MOS\_OOD (↑)    |
| -------------- | --------------- | --------------- |
| Ground Truth   | 3.81 ± 0.09     | 3.70 ± 0.11     |
| StyleTTS2      | 3.83 ± 0.08     | 3.87 ± 0.08     |
| JETS           | 3.57 ± 0.10     | 3.21 ± 0.12     |
| VITS           | 3.44 ± 0.10     | 3.21 ± 0.11     |
| **MVC (Ours)** | **3.87 ± 0.07** | **3.88 ± 0.09** |

### Table 3: Objective Metrics on LJSpeech

| Model          | F0 RMSE (↓)       | MCD (↓)         | WER (↓)   | RTF (↓)    |
| -------------- | ----------------- | --------------- | --------- | ---------- |
| VITS           | 0.667 ± 0.011     | 4.97 ± 0.09     | 7.23%     | 0.0211     |
| StyleTTS2      | 0.651 ± 0.013     | **4.93 ± 0.06** | **6.50%** | 0.0185     |
| **MVC (Ours)** | **0.653 ± 0.014** | 4.91 ± 0.07     | 6.52%     | **0.0177** |

## 🛠️ Troubleshooting

* **NaN Loss:** Ensure the batch size is properly set (e.g., 16 for stable training).
* **Out of Memory:** Reduce batch size or sequence length if OOM errors occur.
* **Audio Quality Issues:** Fine-tune model hyperparameters for specific datasets.

## 📄 License

This project is released under the MIT License. See the LICENSE file for more details.

## 🙌 Contributing

We welcome contributions! Please read the CONTRIBUTING.md file for guidelines on code style, pull requests, and community support.

## 🤝 Acknowledgements

MVC builds on prior work from the Mamba, StyleTTS2, and VITS communities. We thank the authors for their foundational contributions to the field of TTS.

## 📫 Contact

For questions or collaboration, please reach out via GitHub issues or contact us directly at [skumar4@mail.yu.edu](mailto:skumar4@mail.yu.edu).
