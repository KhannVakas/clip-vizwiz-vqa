# clip-vizwiz-vqa
Answering questions about images using CLIP features + PyTorch — includes an answerability detector for unanswerable questions (VizWiz dataset)
# CLIP-Based Visual Question Answering on VizWiz 🔍❓

A Visual Question Answering (VQA) system that answers natural-language questions
about images. Built with OpenAI's CLIP for feature extraction and PyTorch for
classification. Also predicts whether a question is answerable (important for
real-world, low-quality photos from visually impaired users).

## ✨ Features
- 🖼️ Answers questions like *"What color is this?"* or *"What do you see?"*
- 🚫 **Answerability detection** — says "unanswerable" for blurry/unusable images
- ⚡ CLIP (RN50) image+text embeddings → lightweight MLP heads
- 🛡️ Early stopping + ReduceLROnPlateau for stable training

## 📊 Results
| Model | Train | Validation | Test |
|-------|-------|------------|------|
| Answer Predictor | 70.35% | 67.49% | 61.67% |
| Answerability (AP) | 97.89% | 94.46% | 95.51% |

## 🧠 Architecture
Image ──→ CLIP Visual Encoder ─┐
├─→ concat(2048) ─→ MLP ─→ Answer
Question ─→ CLIP Text Encoder ─┘                     └─→ MLP ─→ Answerable? (0–1)


## 🚀 Quick Start
1. Download the [VizWiz 2023 dataset](https://vizwiz.org/) (or the Kaggle mirror)
2. Open `clip-vqa.ipynb` on Kaggle (GPU recommended — works great on free T4)
3. Run all cells — features are cached as `.pth` files automatically

## 📦 Requirements
- Python 3.12, PyTorch, torchvision, CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- scikit-learn, pandas, PIL

## 📁 Files
| File | Purpose |
|------|---------|
| `clip-vqa.ipynb` | Full pipeline: EDA → features → training → inference |

## 📖 How it works
1. **Extract features**: CLIP RN50 encodes each image (1024-d) and question (1024-d) → concatenated to 2048-d
2. **Train two heads**: a 5228-way answer classifier + a binary answerability scorer
3. **Inference**: single forward pass gives both the answer and a confidence score

## 📸 Results Screenshots
<img width="949" height="731" alt="image" src="https://github.com/user-attachments/assets/af59d28d-aa25-4f99-8bf4-6a07c3bb1f53" />
<img width="1028" height="736" alt="image" src="https://github.com/user-attachments/assets/e045c2db-9d15-4207-87c0-e6e9bb32621f" />




## 🙏 Acknowledgements
- [VizWiz dataset](https://vizwiz.org/) — real images from blind photographers
- [OpenAI CLIP](https://github.com/openai/CLIP)

