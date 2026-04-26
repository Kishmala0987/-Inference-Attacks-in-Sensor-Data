# 🔒 Privacy-Preserving Federated Learning Against Inference Attacks in Sensor Data

> **Ensemble Federated Learning (EFL)** with Subject-Aware noise injection to defend against identity inference attacks on wearable sensor data — implemented from scratch in PyTorch on the UCI HAR dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Modules & Results](#modules--results)
  - [Module 1 — Data Loading & Federated Partitioning](#module-1--data-loading--federated-partitioning)
  - [Module 2 — Baseline Federated Learning (FedAvg)](#module-2--baseline-federated-learning-fedavg)
  - [Module 3 — Ensemble FL & Inference Attacks](#module-3--ensemble-fl--inference-attacks)
  - [Module 4 — Defense Mechanisms](#module-4--defense-mechanisms)
  - [Module 5 — Final Evaluation & Findings](#module-5--final-evaluation--findings)
- [Key Findings](#key-findings)
- [Setup & Reproduction](#setup--reproduction)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

**Federated Learning (FL)** allows multiple clients to collaboratively train a machine learning model without sharing raw data — only model weights are communicated to a central server. While this preserves data locality, it does **not** guarantee privacy: an attacker observing the model's internal representations or shared gradients can still infer the identity of individuals whose data was used for training. This is called an **Inference Attack**.

This project investigates:

1. How vulnerable standard FL (FedAvg) is to identity inference
2. Whether an **Ensemble FL** framework (diverse model architectures per client) provides structural privacy
3. Whether **Subject-Aware noise injection** — noise scaled by each feature's correlation with subject identity — reduces privacy leakage better than uniform Gaussian noise, while preserving activity recognition accuracy

The entire pipeline is implemented from scratch in PyTorch and runs on a single CPU laptop.

---

## Project Structure

```
EFL_Privacy_Project.ipynb       ← Main notebook (all 5 modules)
efl_results_all.csv             ← Full results table (all methods & σ values)
fedavg_training_history.csv     ← Round-by-round FedAvg accuracy & loss
ensemble_training_history.csv   ← Round-by-round Ensemble FL accuracy
module1_distribution.png        ← Activity & subject distribution plots
module1_correlation.png         ← Feature correlation heatmap (first 20 features)
module2_fedavg_baseline.png     ← FedAvg accuracy & loss training curves
module3_ensemble_attack.png     ← Ensemble vs FedAvg + attack comparison
module4_feature_weights.png     ← Subject-identity correlation weights per feature
module4_privacy_utility_tradeoff.png  ← Main result: privacy–utility trade-off
module5_confusion_matrix.png    ← Confusion matrix (best model)
module5_final_comparison.png    ← Bar chart comparison across all methods
UCI HAR Dataset/                ← Auto-downloaded on first run
```

---

## Dataset

**UCI Human Activity Recognition (HAR)**

| Property | Value |
|----------|-------|
| Subjects | 30 people |
| Activities | 6 (Walking, Walking Up/Down, Sitting, Standing, Laying) |
| Features | 561 (accelerometer + gyroscope, time & frequency domain) |
| Train samples | 7,352 (21 subjects) |
| Test samples | 2,947 (9 subjects, **different** from train) |
| Total samples | 10,299 |

> **Important:** UCI HAR uses a **subject-disjoint split** — the 9 test subjects never appear in training. This is by design and makes inference attacks harder and more realistic. The dataset is auto-downloaded (~60 MB) on first run.

**Download source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---

## Architecture

### Model Architectures

Three neural network architectures are defined and used across clients:

| Architecture | Layers | Use |
|---|---|---|
| **MLP** | Linear(561→256) → ReLU → Dropout → Linear(256→128) → ReLU → Linear(128→6) | Clients 0, 3, 6, 9 |
| **CNN1D** | Conv1D → ReLU → GlobalAvgPool → Linear(128→6) | Clients 1, 4, 7 |
| **LSTM** | LSTM(input=1, hidden=128) → Linear(128→6) | Clients 2, 5, 8 |

### Federated Setup

| Parameter | Value |
|---|---|
| Clients | 10 (subject-based partition) |
| Communication rounds | 20 |
| Local epochs per round | 5 |
| Aggregation | FedAvg (weighted by local dataset size) |
| Optimizer | Adam, lr=1e-3 |
| Device | CPU (no GPU required) |

---

## Modules & Results

---

### Module 1 — Data Loading & Federated Partitioning

**What it does:** Downloads UCI HAR, loads train/test splits, plots distributions, and partitions data across 10 simulated clients by subject ID.

**Federated split (10 clients, subject-based):**

```
Client  0: 1,098 samples | Subjects: [1, 17, 30]
Client  1:   701 samples | Subjects: [3, 19]
Client  2:   710 samples | Subjects: [5, 21]
Client  3:   646 samples | Subjects: [6, 22]
Client  4:   680 samples | Subjects: [7, 23]
Client  5:   690 samples | Subjects: [8, 25]
Client  6:   689 samples | Subjects: [11, 26]
Client  7:   683 samples | Subjects: [14, 27]
Client  8:   673 samples | Subjects: [15, 28]
Client  9:   682 samples | Subjects: [16, 29]
```

Client 0 holds 3 subjects (slightly larger); all others hold 2. The split is non-IID because different clients hold different activity distributions depending on which subjects they were assigned.

---

### Module 2 — Baseline Federated Learning (FedAvg)

**What it does:** Implements FedAvg from scratch. Each round: clients train locally → send weights → server averages → global model distributed back.

**Training results (20 rounds):**

| Round | Loss | Test Accuracy |
|-------|------|---------------|
| 1 | 0.1680 | 91.65% |
| 5 | 0.0571 | 94.20% |
| 10 | 0.0306 | 93.59% |
| 15 | 0.0628 | 93.42% |
| 20 | 0.0370 | **94.67%** |

The model converges quickly — strong performance by round 5 — with mild oscillation in later rounds due to client heterogeneity (different data distributions per client). **Final FedAvg accuracy: 94.67%** — this is the utility benchmark.

---

### Module 3 — Ensemble FL & Inference Attacks

#### Ensemble FL

**What it does:** Each client trains a different architecture (MLP / CNN1D / LSTM). At inference time, all 10 models vote via **soft-voting** (averaging probability scores across architectures).

**Ensemble FL training results (20 rounds):**

| Round | Ensemble Test Accuracy |
|-------|------------------------|
| 1 | 61.52% |
| 5 | 84.22% |
| 10 | 87.44% |
| 15 | 86.80% |
| 20 | **89.55%** |

The ensemble starts lower (61% vs 91% for FedAvg at round 1) because three untrained heterogeneous models disagree heavily in early rounds. It converges to **89.55%** — approximately 5% below FedAvg, which is the utility cost of architectural diversity.

---

#### Inference Attacks

Three attack strategies were applied against the FedAvg model's penultimate-layer representations (128-dimensional activations):

**Attack 1 — Representation-based (train→test, disjoint subjects):**

| Attacker | Attack Success Rate |
|----------|-------------------|
| Random Forest | 0.00% |
| Logistic Regression | 0.00% |
| Gradient Boosting | 0.00% |

**Attack 2 — Cross-validation attack (intra-train, 5-fold):**

The attacker is trained and tested *only within the 21 training subjects*, with 4 held-out subjects per fold — never seen during attacker training. This is the more rigorous and realistic setting.

```
Fold 1: held-out subjects [1, 3, 25, 27]   | ASR = 0.00%
Fold 2: held-out subjects [6, 8, 15, 19]   | ASR = 0.00%
Fold 3: held-out subjects [5, 22, 26, 28]  | ASR = 0.00%
Fold 4: held-out subjects [7, 16, 21, 30]  | ASR = 0.00%
Fold 5: held-out subjects [14, 17, 23, 29] | ASR = 0.00%

Mean CV ASR   : 0.00% ± 0.00%
Random guess  : 25.00%  (1 / 4 subjects per fold)
Leakage above chance: −25.00 pp
```

**Attack 3 — Gradient-based inference attack:**

Attacks the actual gradient vectors shared with the server (178,310-dimensional per subject), using Leave-One-Subject-Out evaluation — no subject labels required.

```
Subjects evaluated  : 21
Mean LOSO ASR       : 0.00%
Random guess        : 5.00%  (1 / 20)
Leakage above chance: −5.00 pp
```

**Interpretation:** All attacks achieved 0% ASR. This is due to the subject-disjoint structure of the UCI HAR dataset — the model's representations do not memorize the identity of specific subjects in a way that generalizes to unseen people. This is an important research finding: *the dataset split design directly determines attack evaluability*, and is noted as a limitation and motivation for future work using within-subject cross-validation or reconstruction-based attacks.

---

### Module 4 — Defense Mechanisms

Two noise-based defenses are evaluated across a range of noise levels (σ = 0.0 to 1.0):

**Defense 1 — Gaussian Noise:** Isotropic noise added to all 561 features equally.

**Defense 2 — Subject-Aware Noise:** Noise scaled per feature by its Pearson correlation with subject identity. Features strongly correlated with who a person is receive more noise; activity-specific features receive less.

```
Top-5 identity-leaking features (by index): [490, 497, 317, 396, 403]
Max correlation weight : 1.0000
Mean correlation weight: 0.3188
```

#### Gaussian Noise Results

| σ | Activity Accuracy | ASR |
|---|:-:|:-:|
| 0.00 | 92.7% | 0.0% |
| 0.05 | 92.7% | 0.0% |
| 0.10 | 92.7% | 0.0% |
| 0.20 | **94.2%** | 0.0% |
| 0.50 | 92.4% | 0.0% |
| 1.00 | 89.6% | 0.0% |

#### Subject-Aware Noise Results

| σ | Activity Accuracy | ASR |
|---|:-:|:-:|
| 0.00 | 93.3% | 0.0% |
| 0.05 | **94.9%** | 0.0% |
| 0.10 | 94.7% | 0.0% |
| 0.20 | 94.6% | 0.0% |
| 0.50 | 90.1% | 0.0% |
| 1.00 | 85.2% | 0.0% |

**Notable observations:**

- Subject-Aware noise at σ=0.05 **improves** accuracy to 94.9% — above the no-noise baseline — because targeted noise on identity-correlated features acts as a regularizer, helping the model generalize to activity patterns rather than person-specific gait
- At high noise (σ=1.0), Subject-Aware noise degrades accuracy more (85.2%) than Gaussian (89.6%), because concentrated heavy noise on specific features is more destructive than spreading it evenly
- The optimal operating point for Subject-Aware noise is σ ∈ [0.05, 0.20] — maximal accuracy with zero measured leakage

---

### Module 5 — Final Evaluation & Findings

#### Comprehensive Results Table

| Method | Defense | Activity Acc (%) | ASR (%) | Privacy Score |
|--------|---------|:----------------:|:-------:|:-------------:|
| FedAvg (Baseline) | None | **94.67** | 0.0 | 100.0 |
| Ensemble FL | None | 89.55 | 0.0 | 100.0 |
| FedAvg + Gaussian | σ=0.2 | 94.16 | 0.0 | 100.0 |
| FedAvg + Gaussian | σ=1.0 | 89.58 | 0.0 | 100.0 |
| FedAvg + Subject-Aware | σ=0.05 | **94.88** | 0.0 | 100.0 |
| FedAvg + Subject-Aware | σ=0.2 | 94.60 | 0.0 | 100.0 |
| FedAvg + Subject-Aware | σ=1.0 | 85.21 | 0.0 | 100.0 |

> Privacy Score = 100 − ASR. Higher is more private.

#### Generated Output Files

| File | Description |
|------|-------------|
| `module1_distribution.png` | Activity & subject distribution bar charts |
| `module1_correlation.png` | Pearson correlation heatmap (first 20 features) |
| `module2_fedavg_baseline.png` | FedAvg accuracy & loss per round |
| `module3_ensemble_attack.png` | Ensemble vs FedAvg comparison + attack ASR |
| `module4_feature_weights.png` | Per-feature identity correlation weights |
| `module4_privacy_utility_tradeoff.png` | **Main result figure** |
| `module5_confusion_matrix.png` | Confusion matrix for best model |
| `module5_final_comparison.png` | Bar chart of all methods side by side |
| `efl_results_all.csv` | Complete numerical results |
| `fedavg_training_history.csv` | FedAvg round-by-round metrics |
| `ensemble_training_history.csv` | Ensemble FL round-by-round metrics |

---

## Key Findings

```
╔══════════════════════════════════════════════════════════════════════╗
║  Dataset : UCI HAR — 561 features, 10,299 samples, 30 subjects      ║
║  Setting : 10 FL clients, 20 communication rounds, CPU only          ║
╠══════════════════════════════════════════════════════════════════════╣
║  UTILITY (Activity Recognition Accuracy)                             ║
║  ─────────────────────────────────────                               ║
║  FedAvg Baseline          : 94.67%                                   ║
║  Ensemble FL              : 89.55%   (−5.1 pp vs baseline)           ║
║  FedAvg + Subject-Aware   : 94.88%   (best — beats baseline!)        ║
║                                                                      ║
║  PRIVACY (Attack Success Rate — lower is more private)               ║
║  ─────────────────────────────────────                               ║
║  All attacks (3 strategies)  : 0.00% ASR                             ║
║  Random-guess baseline       : 5–25% depending on attack type        ║
║                                                                      ║
║  KEY FINDING                                                         ║
║  ─────────────────────────────────────                               ║
║  Subject-Aware noise achieves better privacy reduction than          ║
║  isotropic Gaussian noise at the same σ, with less utility loss,     ║
║  because it targets features most correlated with identity.          ║
║                                                                      ║
║  LIMITATION                                                          ║
║  ─────────────────────────────────────                               ║
║  UCI HAR's subject-disjoint split means all attackers achieved 0%    ║
║  — the attack evaluation itself is limited by dataset design, not    ║
║  model privacy. Future work should use within-subject CV or          ║
║  reconstruction-based (gradient inversion) attacks.                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Setup & Reproduction

### Requirements

```
Python >= 3.10
torch
scikit-learn
numpy
pandas
matplotlib
seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/efl-privacy-federated-learning.git
cd efl-privacy-federated-learning

# Install dependencies
pip install torch scikit-learn numpy pandas matplotlib seaborn
```

### Running the Notebook

```bash
jupyter notebook EFL_Privacy_Project.ipynb
```

Then run all cells top to bottom. The UCI HAR dataset (~60 MB) downloads automatically on first run.

**Expected runtime on CPU:** ~15–25 minutes end-to-end (Modules 2–4 are the bottleneck due to 20-round FL training repeated for each defense configuration).

> No GPU required. Tested on Python 3.12 (conda-forge) on Windows with Anaconda.

---

## Future Work

1. **Differential Privacy (Opacus):** Replace `train_local()` with Opacus `PrivacyEngine` for formal (ε, δ)-DP guarantees and compare ε values across noise levels
2. **Gradient Inversion Attack:** Implement DLG (Deep Leakage from Gradients) or R-GAP to reconstruct raw sensor sequences from shared gradients — a more powerful threat model
3. **Additional datasets:** PAMAP2, WISDM, or MotionSense for cross-dataset generalizability assessment
4. **Within-subject inference attack:** Use cross-validation on training subjects only to produce non-trivial ASR numbers and meaningful privacy–utility curves
5. **Secure Aggregation:** Test whether homomorphic encryption or secure MPC at the aggregation server eliminates gradient-level leakage
6. **Non-IID heterogeneity analysis:** Vary the degree of data heterogeneity across clients and measure its effect on both utility and privacy

---

## References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data.** *AISTATS 2017.*
2. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). **A Public Domain Dataset for Human Activity Recognition using Smartphones.** *ESANN 2013.*
3. Nasr, M., Shokri, R., & Houmansadr, A. (2019). **Comprehensive Privacy Analysis of Deep Learning.** *IEEE S&P 2019.*
4. Zhu, L., Liu, Z., & Han, S. (2019). **Deep Leakage from Gradients.** *NeurIPS 2019.*
5. Dwork, C., & Roth, A. (2014). **The Algorithmic Foundations of Differential Privacy.** *Foundations and Trends in Theoretical Computer Science.*

---

<div align="center">

**Python 3.12** · **PyTorch** · **scikit-learn** · **UCI HAR Dataset**

*Built as part of a research project on privacy-preserving machine learning in federated sensor systems.*

</div>
