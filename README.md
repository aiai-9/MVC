# MambaVoiceCloning (MVC) - Efficient and Expressive TTS with State-Space Modeling

MambaVoiceCloning (MVC) is an efficient, end-to-end text-to-speech (TTS) framework that eliminates attention and recurrence through selective state-space modeling (SSMs). Built on the principles of efficient sequence modeling, MVC achieves state-of-the-art naturalness, prosody, and latency with significantly reduced computational overhead, making it ideal for real-time and edge applications.

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
2. Preprocess the data:

```bash
python scripts/preprocess.py --dataset LJSpeech --output_path data/LJSpeech
```

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

## 📊 Evaluation

Run objective and subjective evaluations using provided scripts:

```bash
python evaluate.py --config_path ./configs/config.yml
```

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
