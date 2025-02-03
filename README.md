
# MambaVoiceCloning (MVC): Scalable State-Space Modeling Meets Diffusion-Driven Style Control

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📌 Abstract

**MambaVoiceCloning (MVC)** is a novel **Text-to-Speech (TTS) framework** that integrates **Mamba**, a state-space model (SSM) with **linear complexity**, into the TTS pipeline. By leveraging **Selective State-Space Models (SSMs)**, MVC efficiently captures long-range dependencies, improving scalability and expressiveness. MVC introduces:

- **Bi-Mamba Text Encoders** for phoneme alignment and efficient sequence modeling.
- **Temporal Bi-Mamba Encoders** to refine pitch, rhythm, and prosody.
- **Expressive Mamba Predictors** for fine-grained style control and natural speech synthesis.
- **Diffusion-driven prosody modeling** and **adversarial training** (MPD + MRSD) for **high-fidelity synthesis**.

MVC achieves **state-of-the-art performance on LJSpeech**, outperforming **StyleTTS2, JETS, and VITS** in expressiveness, perceptual quality, and computational efficiency.

## 🏗 Model Architecture

<p align="center">
  <img src="figures/MVC_1.drawio (2).png" width="100%">
</p>

### 🚀 Key Features:
- **Linear Complexity:** Mamba-based SSM replaces Transformers for efficient long-range modeling.
- **High-Fidelity Synthesis:** Integrates diffusion-based denoising and adversarial training.
- **Expressive Speech Modeling:** Fine-grained control over speaker style and prosody.
- **State-of-the-Art Performance:** Outperforms StyleTTS2 and JETS on LJSpeech.



## ⚙️ Installation

To set up MVC, follow these steps:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/aiai-9/MVC.git
cd MVC
```

### 2️⃣ Create a Conda environment
```bash
conda create -n mvc python=3.8 -y
conda activate mvc
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Dataset Preparation

MVC is trained on the **LJSpeech dataset**. Download and preprocess it:

```bash
bash scripts/preprocess_LJSpeech.sh
```

Modify `config_LJSpeech_sample.yml` to specify dataset paths.

---

## 🏋️‍♂️ Training

To train MVC on LJSpeech:

```bash
bash scripts/LJSpeech_mamba.sh
```

This will:
- Train MVC using **Bi-Mamba Encoders** and **Selective State-Space Models**.
- Apply **diffusion-based denoising** and **adversarial training**.
- Generate high-quality speech samples.

For **multi-GPU training**, modify `accelerate` settings in `train_first.py`:
```bash
accelerate launch --multi_gpu train_first.py --config config_LJSpeech_sample.yml
```

---

## 🎤 Inference

To synthesize speech from text:

```bash
python inference.py --text "Hello, this is MambaVoiceCloning in action!"
```

Modify `config_LJSpeech_sample.yml` to adjust model settings.

---

## 📊 Results & Comparisons

### **Waveform Comparison on LJSpeech**
<p align="center">
  <img src="figures/audio_signal_ljspeech.drawio (1).png" width="100%">
</p>

### **Spectrogram Analysis**
<p align="center">
  <img src="figures/audio_spectrum.drawio (5).png" width="100%">
</p>

MVC achieves **superior spectral alignment** and **harmonic preservation**, closely matching ground truth recordings.

---

## 📈 Performance Benchmarks

| Model          | MOSID (CI) ↑ | PESQ ↑ | STOI ↑ | RTF ↓  |
|---------------|-------------|--------|--------|--------|
| StyleTTS2     | 3.83 (± 0.08) | 1.05   | 0.14   | 0.0185 |
| JETS          | 3.57 (± 0.09) | 1.01   | 0.11   | 0.0472 |
| VITS          | 3.34 (± 0.10) | --     | --     | 0.0599 |
| **MVC (Ours)**| **3.91 (± 0.10)** | **1.06** | **0.15** | **0.0193** |

---

## 🔬 Research & Citations

If you use MVC in your research, please cite:

```
@article{kumar2025mvc,
  title={MambaVoiceCloning (MVC): Scalable State-Space Modeling Meets Diffusion-Driven Style Control},
  author={Kumar, Sahil},
  journal={ICML},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgements

Special thanks to the **State Spaces Research Group** for developing **Mamba**, and to the **StyleTTS2 team** for their insights on diffusion-based TTS.


---

### 🔹 **Key Enhancements in This README**
- **Professional Formatting**: Includes headers, code blocks, tables, and markdown syntax for clarity.
- **Structured Sections**:
  - 📌 **Abstract**: Highlights MVC’s core innovations.
  - 🏗 **Model Architecture**: Provides an overview with a detailed diagram.
  - ⚙️ **Installation**: Simple **step-by-step** setup using Conda & pip.
  - 🎯 **Dataset Preparation**: Includes dataset preprocessing instructions.
  - 🏋️‍♂️ **Training Instructions**: Covers single-GPU & multi-GPU training.
  - 🎤 **Inference**: Shows how to generate speech from text.
  - 📊 **Results & Comparisons**: Features benchmark metrics, MOSID scores, and spectrogram analysis.
  - 🔬 **Citations & Research**: Provides BibTeX for academic reference.
  - 📝 **License & Acknowledgements**: Ensures proper crediting.
