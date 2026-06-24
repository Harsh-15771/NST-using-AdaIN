# Neural Style Transfer using AdaIN

A web-based **Neural Style Transfer** project built with **PyTorch** and **Flask**, using **Adaptive Instance Normalization (AdaIN)** for **arbitrary style transfer**.  
The project allows users to upload a **content image** and a **style image**, control the **style strength**, and generate a stylized output image through a simple browser interface.

In addition to the web app, the repository also contains the **full training pipeline** used to train the style transfer decoder using **COCO 2017** as the content dataset and **Painter by Numbers** as the style dataset.

---

# Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [How AdaIN Works](#how-adain-works)
- [Project Structure](#project-structure)
- [Web App Workflow](#web-app-workflow)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [How to Run the Web App](#how-to-run-the-web-app)
- [How to Train the Model](#how-to-train-the-model)
- [Checkpointing and Resume Support](#checkpointing-and-resume-support)
- [Challenges Faced](#challenges-faced)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Author](#author)

---

# Project Overview

Neural Style Transfer (NST) is a deep learning technique that generates an image that preserves the **content** of one image while adopting the **style** of another. Traditional optimization-based style transfer methods are often slow because they solve an optimization problem separately for every image pair.

This project uses **Adaptive Instance Normalization (AdaIN)**, a feed-forward approach for **arbitrary style transfer** that performs stylization in a single forward pass.

The repository contains **two major parts**:

## 1. Web Application

A Flask-based web interface where users can:

- upload a **content image**
- upload a **style image**
- choose the **alpha value** to control stylization strength
- generate and preview the stylized output

## 2. Training Pipeline

A PyTorch training pipeline for training the AdaIN decoder using:

- **COCO 2017** as the content dataset
- **Painter by Numbers** as the style dataset
- staged training at **256×256** and **512×512**

---

# Features

- **Arbitrary neural style transfer** using AdaIN
- **Flask web app** for interactive style transfer
- Upload **content and style images** directly from the browser
- Control style transfer intensity using **alpha blending**
- **PyTorch training pipeline** for training and fine-tuning the decoder
- **Checkpoint-based resume training**
- Supports **high-resolution fine-tuning**
- Clean project structure separating:
  - training code
  - model utilities
  - web app logic
  - example assets
  - uploaded/generated images

---

# How AdaIN Works

Adaptive Instance Normalization transfers style by matching the **channel-wise mean and standard deviation** of the content feature map to those of the style feature map.

If:

- \(x\) = content feature map
- \(y\) = style feature map

then AdaIN is defined as:

\[
\text{AdaIN}(x, y) = \sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right) + \mu(y)
\]

where:

- \(\mu(\cdot)\) = channel-wise mean
- \(\sigma(\cdot)\) = channel-wise standard deviation

This transforms the content representation so that it preserves **content structure** while adopting **style statistics** from the style image.

---

# Project Structure

```bash
NST using AdaIN/
│
├── content_data/              # content images used for training
├── examples/                  # example images shown in the website demo
├── experiment/                # saved training checkpoints and experiment outputs
├── static/
│   └── uploads/               # uploaded content/style images + generated outputs
├── style_data/                # style images used for training
├── templates/
│   └── index.html             # main web app page
│
├── utils/
│   ├── models.py              # VGG encoder and decoder architecture
│   └── utils.py               # AdaIN, transforms, dataset utilities, helper functions
│
├── app.py                     # Flask web app for inference
├── Procfile.txt               # deployment configuration
├── requirements.txt           # project dependencies
├── train.py                   # model training script
├── vgg_normalised.pth         # pretrained VGG weights
└── README.md
```

---

# Web App Workflow

The web application provides an interface for performing style transfer directly in the browser.

## User flow

1. Upload a **content image**
2. Upload a **style image**
3. Set the **alpha value**
4. Click **Transfer Style**
5. View the stylized output image

## What happens internally

Inside `app.py`, the application:

1. loads the pretrained **VGG encoder**
2. loads the trained **decoder checkpoint**
3. resizes both input images to **512 × 512**
4. extracts content and style features
5. applies **Adaptive Instance Normalization**
6. blends the transformed feature map with the original content features using the chosen **alpha**
7. decodes the stylized feature map into an output image
8. saves the result inside `static/uploads/`

---

# Model Architecture

The project follows the standard AdaIN pipeline:

## 1. Encoder

A pretrained **VGG-based encoder** extracts deep features from:

- the content image
- the style image

The encoder remains fixed during training and inference.

## 2. Adaptive Instance Normalization (AdaIN)

The deepest content feature map is normalized using the style feature statistics, creating a target representation that combines:

- the structure of the content image
- the style characteristics of the style image

## 3. Decoder

A trainable decoder reconstructs an RGB image from the AdaIN-transformed feature representation.

## 4. Alpha Blending

The web app supports an **alpha parameter** that controls the stylization strength:

- `alpha = 1.0` → stronger stylization
- smaller alpha values → more content preservation

The transformed feature map is blended as:

\[
t = \alpha \cdot \text{AdaIN}(c, s) + (1-\alpha)\cdot c
\]

where:

- \(c\) = content features
- \(s\) = style features

---

# Training Strategy

The decoder was trained in a **two-stage setup**:

## Stage 1 — Base Training at 256×256

- **Image size:** `256`
- **Batch size:** `16`
- **Style weight:** `5`

This stage is used to learn stable stylization efficiently at a lower resolution.

## Stage 2 — Fine-tuning at 512×512

- **Image size:** `512`
- **Batch size:** `8`
- **Style weight:** `10`

This stage improves output quality at higher resolution and helps refine finer stylistic details.

## Why staged training?

Training directly at high resolution is expensive and memory-intensive.  
A staged 256 → 512 pipeline:

- improves training stability
- reduces GPU memory pressure in early training
- speeds up convergence
- allows later high-resolution refinement

---

# Datasets

## Content Dataset

**COCO 2017** was used as the content dataset.

Example Kaggle path used during training:

```bash
/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/test2017
```

## Style Dataset

**Painter by Numbers** was used as the style dataset.

During training, style images were taken from the Painter by Numbers dataset and used for style sampling.

---

# Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Flask**
- **Flask-WTF**
- **Flask-Bootstrap**
- **PIL**
- **NumPy**
- **Kaggle Notebooks / GPU training**

---

# How to Run the Web App

## 1. Clone the repository

```bash
git clone https://github.com/Harsh-15771/NST-using-AdaIN.git
cd "NST using AdaIN"
```

## 2. Create and activate a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Make sure the required model files are available

You need:

- `vgg_normalised.pth`
- a trained decoder checkpoint (for example `decoder_final.pth`)

Update the decoder checkpoint path inside `app.py` if needed.

## 5. Run the Flask app

```bash
python app.py
```

## 6. Open the app in the browser

Go to:

```bash
http://localhost:5000
```

---

# How to Train the Model

The training pipeline is implemented in `train.py`.

## Example training command

```bash
python train.py \
  --content_image /path/to/content_dataset \
  --style_image /path/to/style_dataset \
  --vgg /path/to/vgg_normalised.pth \
  --experiment experiment_name \
  --epochs 10 \
  --batch_size 16 \
  --final_size 256 \
  --style_weight 5
```

## Example resume training command

```bash
python train.py \
  --content_image /path/to/content_dataset \
  --style_image /path/to/style_dataset \
  --vgg /path/to/vgg_normalised.pth \
  --experiment experiment_name \
  --epochs 10 \
  --start_epoch 120 \
  --batch_size 16 \
  --final_size 256 \
  --style_weight 5 \
  --resume \
  --decoder_path /path/to/decoder_epoch_120.pth \
  --optimizer_path /path/to/optimizer_epoch_120.pth
```

## Example 512 fine-tuning command

```bash
python train.py \
  --content_image /path/to/content_dataset \
  --style_image /path/to/style_dataset \
  --vgg /path/to/vgg_normalised.pth \
  --experiment experiment_name \
  --epochs 3 \
  --start_epoch 150 \
  --batch_size 8 \
  --final_size 512 \
  --style_weight 10 \
  --resume \
  --decoder_path /path/to/decoder_epoch_150.pth \
  --optimizer_path /path/to/optimizer_epoch_150.pth
```

---

# Checkpointing and Resume Support

The training script supports:

- saving **decoder checkpoints**
- saving **optimizer checkpoints**
- resuming from any saved checkpoint

Resume training is controlled using:

- `--resume`
- `--decoder_path`
- `--optimizer_path`
- `--start_epoch`

This was especially useful for long Kaggle training runs where training had to be split into multiple chunks.

---

# Challenges Faced

Some practical challenges encountered during the project:

- training large style transfer models under **Kaggle runtime limits**
- resuming checkpoints correctly when using **DataParallel**
- balancing **batch size vs GPU memory** at 512 resolution
- managing long-running checkpointed training jobs
- reducing logging/output overhead for stable training runs
- integrating the trained model into a web application for inference

These issues were handled through:

- staged training
- checkpoint-based resume runs
- controlled save intervals
- multi-stage fine-tuning
- separating training and inference logic cleanly

---

# Results

The project produces stylized images that:

- preserve the semantic structure of the content image
- adopt artistic texture and color statistics from the style image
- can be generated in a **single forward pass**

## Demo examples

The repository includes an `examples/` folder used by the website to showcase example stylizations and demonstrate how the system works.

## Output storage

Uploaded input images and generated stylized images are saved in:

```bash
static/uploads/
```

---

# Future Improvements

Possible future improvements for the project:

- replace `DataParallel` with **DistributedDataParallel (DDP)** for better multi-GPU efficiency
- optimize high-resolution training further
- allow users to select from **preset example styles**
- add a **download button** for generated outputs
- add **style interpolation** between multiple style images
- deploy the web app online
- support **video style transfer**
- improve frontend UI and add better preview controls

---

# References

**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**  
Xun Huang, Serge Belongie  
ICCV 2017

Paper: https://arxiv.org/abs/1703.06868

---

# Author

**Harsh Mishra**  
Final-year Engineering Undergraduate  
Interested in **Machine Learning, Deep Learning, and Full Stack Development**
