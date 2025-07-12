# Exoplanet-Transit CNN Classifier 📉🚀
Goal: train a Convolutional Neural Network that tells real / synthetic exoplanet transits apart from other variable-star light-curves (binary, pulsating, “normal” spotted stars).

All light-curves are turned into minimalist 128 × 128-pixel PNGs and stored in a clean folder hierarchy so you can re-generate data, (re-)train, and run inference with a single notebook.

# 🌐 Project layout
├── light_curves/                 ← all images live here
│   ├── binary_star/              ← detached eclipsing binaries
│   ├── pulsating_star/           ← Cepheid-like pulsators
│   ├── normal_star/              ← spot / granulation noise
│   ├── exoplanet/                ← synthetic & real transits
│   └── tests/                    ← hold-out set for final score
│
├── lightcurve_generator.py       ← low-level synthetic curves API  ◆
├── images.py                     ← batch generator / noise add-on  ◆
├── real_exoplanets.py            ← downloads real TESS/Kepler data ◆
│
├── primer_modelo.ipynb           ← ✨ Jupyter notebook: builds & trains the CNN
└── README.md                     ← you are here
# 🔧 Quick setup
####  1) clone & create an isolated env
git clone <your-repo>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate  # Win: .venv\Scripts\activate

### 2) install requirements
pip install -r requirements.txt
  or, bare minimum:
pip install tensorflow matplotlib numpy pillow
pip install lightkurve astroquery pandas tqdm         # for real data

It'rs recommended you use a package manager like anaconda since tensorflow doesnt work with every python version(the one used for the project was 3.11)
And there could be other problems with dependencies.
# 🏗️ Generate / download the data (one-off)
| Task                 | Command                     | What it does                                                                                                                                                                                                                     |
| -------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Synthetic curves** | `python images.py`          | • wipes old images in `binary_star/ pulsating_star/ normal_star/` <br>• creates 1 100 light-curves per class with Gaussian noise (σ ≈ 1.5 e-3) and resizes every PNG to **128×128**.                                             |
| **Real transits**    | `python real_exoplanets.py` | • queries NASA Exoplanet Archive, <br>• downloads the first available Kepler/K2/TESS **PDCSAP** light-curve, normalises it, <br>• draws an axis-free red line and saves into **`light_curves/tests/`** for unbiased evaluation.  |
# 🧠 Training the CNN
Open primer_modelo.ipynb and simply run all cells:

Data loader: scans light_curves/* and builds tf.data.Dataset pipelines with on-the-fly augmentation (random flips, small jitter).

Model:
Input (128,128,1) → Conv-ReLU-MaxPool ×2 → GlobalAveragePooling2D
                   → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)
Global average pooling keeps the network resolution-agnostic, so you won’t have to tweak Dense sizes when changing the image size.

* Training: binary cross-entropy, Adam (lr = 1 e-3), early-stopping on val_loss.

* Save: writes exoplanet_cnn_weights.h5 next to the notebook.

* Evaluation: loads the model and scores it on light_curves/tests/ (accuracy, ROC-AUC, confusion matrix).