

# Deep MIML: Multi-Instance Multi-Label Image Classification

Deep learning framework for Multi-Instance Multi-Label (MIML) classification using PyTorch. Each input is a **bag** of images, and the model predicts **multiple labels** indicating which object categories are present across the bag.

## Overview

In standard classification, one image maps to one label. MIML generalizes this: a bag contains a variable number of images (instances) from different categories, and the model must predict which categories are represented. This is useful for problems like scene understanding, medical image analysis, and any domain where inputs are naturally grouped and have multiple simultaneous labels.

This project constructs synthetic MIML datasets from CIFAR-10 and trains deep CNN-based models with two instance-aggregation strategies:

- **Average Pooling** — extracts features from each instance with a pretrained CNN backbone, averages them across the bag, and classifies the aggregated representation.
- **Attention Pooling** — learns per-class attention weights over instances, producing a weighted combination of features before classification. Each class gets its own attention head, allowing the model to focus on different instances for different labels.

## Project Structure

```
deep_miml/
├── cifar_bags.py        # MIML bag construction from CIFAR-10
├── models.py            # Average and Attention model architectures
├── train.py             # Training loop with early stopping
├── test.py              # Evaluation across saved models
├── create_dataset.py    # Dataset generation and serialization
├── run_experiments.py   # Batch training across model configurations
├── plot_results.py      # Result visualization
├── imagenet_bags.py     # ImageNet data loader (experimental)
├── utils.py             # Precision/recall metrics
└── __init__.py
notebooks/               # Exploratory analysis and result plotting
```

## Setup

```bash
pip install -e .
```

**Dependencies:** Python >= 3.6, PyTorch, torchvision, numpy, tqdm, scikit-learn, pandas, seaborn, matplotlib

## Usage

### 1. Create the MIML dataset

Constructs bags from CIFAR-10 images. Each bag contains 1 to `num_categories/2` categories, with up to `m` instances per category.

```bash
python -m deep_miml.create_dataset \
    --num_bags_train 2000 \
    --num_bags_test 100 \
    --num_bags_val 100 \
    --m 4 \
    --save_file_to data/miml_data.pt
```

### 2. Train a model

```bash
python -m deep_miml.train \
    --model_name resnet18 \
    --model_type avg \
    --use_pretrained True \
    --data_file_path data/miml_data.pt \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --patience 5
```

**Supported backbones:** `resnet18`, `resnet34`, `resnet50`, `resnext50`, `alexnet`

**Aggregation types:** `avg` (average pooling), `attention` (learned attention)

Add `--cuda` to train on GPU. Models are saved to `--save_folder` (default: `/tmp`).

### 3. Run all experiments

Trains all backbone/aggregation combinations in sequence:

```bash
python -m deep_miml.run_experiments
```

### 4. Evaluate and plot

```bash
python -m deep_miml.test --data_file_path data/miml_test_data_5000.pt --save_folder results/
python -m deep_miml.plot_results
```

## Evaluation

Models are evaluated with **Precision@k** and **Recall@k** for k = 1, ..., 6 on the 10-class CIFAR-10 label space.

### Results (2,000 training bags)

| Model | Type | Device | Time | P@1 | P@3 | R@1 | R@3 |
|---|---|---|---|---|---|---|---|
| ResNeXt-50 | avg | CUDA | 90s | 0.889 | 0.618 | 0.448 | 0.801 |
| ResNeXt-50 | attention | CPU | 402s | 0.860 | 0.598 | 0.422 | 0.768 |
| ResNet-18 | avg | CUDA | 37s | 0.876 | 0.615 | 0.434 | 0.791 |
| ResNet-18 | attention | CPU | 427s | 0.873 | 0.617 | 0.432 | 0.792 |

Average pooling with pretrained backbones performs competitively with attention on this synthetic benchmark, while training significantly faster.

## License

MIT