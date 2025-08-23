# 🏗️ # PINN vs FEM for a 1D Elastic Bar

**Physics-Informed Neural Networks (PINNs) vs a purely data-driven black-box** on a classical solid-mechanics benchmark: the axial displacement of a prismatic bar. The repo provides:
- 🔧 A minimal **FEM reference solver**,
- 🧠 A **PINN** that enforces the PDE and boundary conditions,
- 📊 A **black-box MLP** trained on noisy supervised data,
- 🔬 Reproducible **experiments** (data-efficiency, noise robustness, extrapolation).

## 📐 Problem statement

Consider a straight 1D bar of length $\text(L)$ with axial displacement $u(x)$. Let $E(x)$ be Young's modulus, $A(x)$ cross‑sectional area, and $f(x)$ the axial body force per unit length. The governing equation in strong form is:


$\Large{-\frac{d}{dx}\left(E(x)A(x)\frac{du}{dx}\right) = f(x), \quad x\in(0,L).}$

Supported boundary conditions on $x=0$ and $x=L$:

- 📌 **Dirichlet (displacement)**: $u=u_0$.
- ⚡ **Neumann (traction / tip load)**: $EA\,u'(x)=P$.
- 🔗 **Robin (spring / convective)**: $EA\,u'(x)+h\,u = g$.

> 💡 This project mainly showcases **tip load** and **body force** cases with homogeneous material; heterogeneity and Robin BCs are included in the code and easy to toggle.lastic Bar

---


## 📂 Repository layout

    .
    ├── 🎨 assets/
    ├── 📁 data/
    │   ├── 📊 outputs/                # FEM CSVs, logs, figures
    │   └── 🎯 supervised/             # generated BB datasets
    ├── 💻 src/
    │   ├── ⚙️ core/                   # reusable components
    │   │   ├── fem.py
    │   │   ├── pinn.py
    │   │   ├── physics.py
    │   │   └── utils.py
    │   ├── 🚀 training/
    │   │   ├── run_fem.py             # CLI: make FEM CSV for a case/BCs
    │   │   ├── run_pinn.py            # CLI: train PINN & save log
    │   │   └── train_blackbox.py      # CLI: train supervised MLP & save log
    │   ├── 🔬 experiments/
    │   │   ├── data_gen.py            # create noisy datasets for the Black box model
    │   │   └── sweep_metrics.py       # data-efficiency & noise robustness sweeps
    │   └── 📈 visualization/
    │       ├── plot_metrics.py
    │       └── hero_figure.py
    ├── 🧪 tests/
    ├── ⚡ Makefile                    # one-command pipelines
    ├── 🌍 env.yml
    ├── 📦 pyproject.toml              # packaging metadata
    └── 📖 README.md                   # this file

---

## ⚡ Quickstart (60 seconds)

```bash
    # 0️⃣ Create env
    conda env create -f env.yml
    conda activate py310-torch

    # 1️⃣ In-range demo (tip load P=0.60) → FEM + PINN + BB + joint figure
    make inrange CASE=tip_load P=0.60 EPOCHS=1200 BB_EPOCHS=1200

    # 2️⃣ Extrapolation demo (outside BB training range)
    make extrap               # alias for P=1.20; adjust in Makefile if desired

    # 3️⃣ Generate the hero figure
    make hero
  ```

📁 Results appear under `data/outputs/<CASE>_P<P>/`. Example artifacts:
- `fem_tip_load.csv` – reference solution $(x,u)$
- `pinn_tip_load_trainlog.npz` – PINN losses & snapshots,
- `bb_trainlog.npz` – black-box losses & snapshots,
- `hero.png/.svg`

## What's implemented
- **FEM reference** with linear 1D bar elements.
- **PINN**:
  - fully connected MLP
  - PDE residual + weighted BC losses
  - optional input/output normalization.
- **Black-box MLP**: supervised $[x, P] \rightarrow u$, integrated no pysics.
- **Experiments**:
  - *Data efficiency*: error vs number of samples per configuration $(M)$
  - *Noise robustness*: error vs label noise $\sigma$
  - *Reliability*: $\mu \displaystyle \pm \sigma$ across seeds.
  - *Extrapolation*: evaluation at $P$ beyond dataset range

## 🏆 Selected results

<p align="center">
  <img src="data/outputs/hero_full.png" width="820" alt="Hero figure — tip load (in-range & extrapolation)">
</p>

<p align="center">
  <img src="data/outputs/anim_joint_extrap.gif" width="820" alt="Joint animation — predictions (top) + losses (bottom), tip load extrapolation P=1.20">
  <br/>
  <em></em>
</p>

### 📊 Key Findings:

- ✅ For **interpolation** (in-range), $\text{PINN}$ and $\text{BB}$ achieve low error; $\text{PINN}$ converges in fewer epochs.
- 🚀 For **Extrapolation**, $\text{PINN}$ remains accurate; $\text{BB}$ deteriorates.
- 🛡️ In the case of **noisy data**, $\text{PINN}$ has higher reliability.
- 📈 On a **limited amount of training data** samples, $\text{PINN}$ functions much better than $\text{BB}$.

