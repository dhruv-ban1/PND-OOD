# PND-OOD: Outlier Imagination via Projected Normal Distributions and Diffusion Models

This project explores a probabilistic approach to out-of-distribution (OOD) image generation by modeling the latent space using **Projected Normal Distributions (PND)** and decoding outlier embeddings using **Stable Diffusion**. Our method builds upon the DREAM-OOD framework but replaces the von Mises-Fisher (vMF) modeling with PND to provide richer, uncertainty-aware latent representations.

## 🔍 Overview

Out-of-distribution detection is important for identifying inputs that differ from the training distribution. Instead of relying on real-world outlier datasets, we propose generating synthetic OOD samples by:

1. Learning a latent embedding space where each class is modeled by a Projected Normal Distribution.
2. Sampling low-likelihood latent vectors under this distribution.
3. Using a text-conditioned **Stable Diffusion** model to generate realistic, visually plausible OOD images.

Our method improves the flexibility and interpretability of outlier sampling by incorporating class-specific covariance and probabilistic scoring.

---

## 🚀 Features

- PND-based latent space modeling
- Class-wise uncertainty-aware sampling
- Posterior likelihood estimation
- Integration with Stable Diffusion via custom embedding injection
- Outlier image generation conditioned on embedding-space deviations
- Ready-to-use pipeline for training and sampling

---

## 📂 Project Structure

```bash
project/
│
├── src/                    # Core codebase
│   ├── encoder.py          # ResNet encoder with latent normalization
│   ├── pnd_loss.py         # PND loss function and scoring
│   ├── sampler.py          # Outlier embedding sampler
│   ├── diffusion_hook.py   # Modified Stable Diffusion hook to inject embeddings
│   ├── utils.py            # Helper functions (normalization, metrics, etc.)
│
├── outlier_embeddings.npy  # Precomputed outlier embeddings (optional)
├── generate_images.py      # Generate OOD images using diffusion
├── train_encoder.py        # Train latent space with PND loss
├── requirements.txt        # Required dependencies
├── README.md               # Project overview
