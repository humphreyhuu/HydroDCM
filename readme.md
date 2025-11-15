# HydroDCM: High-Level Implementation Guide

This document gives a concise, method-centric guide for reproducing **HydroDCM**. It specifies data shapes, domain splits, model modules, objectives, and the training–inference flow. Detailed coding steps and configuration are intentionally omitted so another LLM can implement them.

## 1. Data and Notation

**Signals per reservoir (daily):** temperature (T), precipitation (P), inflow (Q).

**Supervised samples (one-step ahead by default):**
Input window $\mathbf{X}\in\mathbb{R}^{T_{\text{in}}\times F_{\text{in}}}$ with $F_{\text{in}}=3$ for $[T,P,Q]$.
Label $y\in\mathbb{R}$ as next-day inflow.
Batch shapes: `X [B, T_in, 3]`, `y [B]`.
For multi-step forecasting, use `y [B, H]`.

**Spatial attributes per reservoir:** $\mathbf{s}\in\mathbb{R}^{3}$ = (longitude, latitude, elevation).
Batch shape aligned with `X`: `s [B, 3]` (repeat per sample of a reservoir).

**Normalization:** z-score inputs using **source-train** statistics; apply to val/test. Standardize $\mathbf{s}$ over reservoirs for stable conditioning.

## 2. Domain Split

Treat **each reservoir as one domain**. Use the common time span **1999–2011** for all 30 reservoirs.
**Target domains** $P_{te}$: **MCR**, **JVR**, **MCP** (isolated stream branches to induce stronger distribution shift).
**Source domains** $P_{tr}$: remaining 27 reservoirs, split at the **reservoir level** into train and validation without leakage.

## 3. Model Overview

HydroDCM has four modules and a predictive head:

* **Feature extractor $f_\phi$**: temporal backbone. Use **ED-LSTM** (encoder–decoder LSTM). Output hidden representation $\mathbf{h} = f_\phi(\mathbf{X})$ with shape `[B, H_feat]`. One-step setting can skip the decoder and use encoder state.
* **Domain encoder (linear projection)**: map spatial meta to a pseudo-domain embedding $\mathbf{v} = \mathbf{W}\mathbf{s}+\mathbf{b} \in \mathbb{R}^d$ (e.g., $d=32$). This supports contrastive separation of reservoirs.
* **Domain discriminator $d_\theta$** with **gradient reversal** on the $\mathbf{h}$ branch to learn **invariant features** $\mathbf{z}$ that suppress domain cues.
* **FiLM adapter $m_\delta$**: produce **scaling** $\gamma(\mathbf{s})\in\mathbb{R}^{H_{\text{feat}}}$ and **shifting** $\delta(\mathbf{s})\in\mathbb{R}^{H_{\text{feat}}}$ from $\mathbf{s}$ via small MLPs (shared trunk with two heads is fine).
* **Predictive head $p_\omega$**: map the modulated features to inflow. One-step: `H_feat → 1`; multi-step: `H_feat → H`.

## 4. Objectives

* **Contrastive loss $L_{\text{con}}$** on pseudo-domain embeddings $\mathbf{v}$ using **InfoNCE** with positives from the same reservoir and negatives from others.
* **Adversarial loss $L_{\text{adv}}$** between $\mathbf{v}$ and $\mathbf{h}$ with a discriminator $d_\theta$; apply a **GRL** to push $f_\phi$ to remove domain information from $\mathbf{h}$, yielding invariant $\mathbf{z}$.
* **Supervised loss $L_{\text{sup}}$** as **MSE** on the prediction from the **modulated** invariant features:

$$
\tilde{\mathbf{z}} = \gamma(\mathbf{s}) \odot \mathbf{z} + \delta(\mathbf{s}),\qquad \hat{y} = p_\omega(\tilde{\mathbf{z}}).
$$
* **Total objective:**

$$
L_{\text{total}} = L_{\text{con}} \;+\; \lambda_{\text{adv}}\, L_{\text{adv}} \;+\; \lambda_{\text{sup}}\, L_{\text{sup}}.
$$

  Choose $\lambda_{\text{adv}}$ and $\lambda_{\text{sup}}$ to balance invariance and accuracy.

## 5. Inference Architecture

**1. Temporal extraction:** compute $\mathbf{h} = f_\phi(\mathbf{X})$.

**2. Invariant feature learning:** project $\mathbf{s}$ to $\mathbf{v}$; optimize **InfoNCE** on $\mathbf{v}$. Train $d_\theta$ to discriminate domain signal while $f_\phi$ with **GRL** suppresses it in $\mathbf{h}$, producing invariant $\mathbf{z}$.

**3. Domain-conditioned modulation:** obtain $\gamma(\mathbf{s})$ and $\delta(\mathbf{s})$ from $m_\delta$; form $\tilde{\mathbf{z}}$ and optimize **MSE** through $p_\omega$. This explicitly integrates spatial meta as hydrological priors, which is necessary because location and terrain govern precipitation–runoff response and differ across reservoirs.

## 6. Inference Flow

Use **only** $f_\phi$, $m_\delta$, and $p_\omega$. For a target reservoir, standardize its input window $\mathbf{X}$ with source stats, standardize $\mathbf{s}$, compute $\mathbf{h}\rightarrow \mathbf{z}$, modulate with $\gamma(\mathbf{s}),\delta(\mathbf{s})$ to get $\tilde{\mathbf{z}}$, then predict $\hat{y}$. Inverse-transform inflow for reporting if needed.
