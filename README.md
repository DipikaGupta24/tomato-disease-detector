# Tomato Leaf Disease Detection + Generative AI Report

An end-to-end project: a CNN (transfer learning on MobileNetV2) classifies a tomato
leaf photo into one of 10 classes (9 diseases + healthy), Grad-CAM shows *why* the
model thinks so, and a generative AI model (Gemini) turns the raw prediction into a
plain-language treatment report.

## What's in this folder

| File | What it's for | Where it runs |
|---|---|---|
| `train_tomato_model.ipynb` | Downloads the dataset, trains the model, evaluates it, saves it | **Google Colab** |
| `app.py` | The user-facing app: upload a leaf photo, get prediction + Grad-CAM + AI report | **Your laptop, then Streamlit Cloud** |
| `requirements.txt` | Python packages the app needs | used by Streamlit Cloud automatically |

## Notebook structure

The notebook now has two parts:

- **Part 1 — Basic Model**: transfer learning on MobileNetV2, frozen base, standard
  augmentation. Gets you to ~90% validation accuracy on PlantVillage's lab-condition
  images. Saves `tomato_model.h5`.
- **Part 2 — Advanced Improvements**: leaf-cropping preprocessing (removes background
  reliance), stronger augmentation, mixing in real-world PlantDoc images during
  training, fine-tuning the base model, and label smoothing for better-calibrated
  confidence. Evaluates on BOTH the original lab validation set and a held-out
  real-world test set. Saves `tomato_model_advanced.h5`.

Run Part 1 completely first, then run Part 2 cells in order (Part 2 depends on
folders created in Part 1). Use `tomato_model_advanced.h5` in the app — it's the
better-performing, more robust model.

## Step-by-step plan

### Part 1 — Train the model (in Google Colab)

1. Go to [colab.research.google.com](https://colab.research.google.com), sign in with
   a Google account.
2. `File → Upload notebook` → upload `train_tomato_model.ipynb`.
3. `Runtime → Change runtime type → GPU (T4)` → Save. This gives you a free GPU,
   training will take ~15-25 minutes instead of hours.
4. Run each cell **top to bottom**, in order. The notebook itself explains each step
   as you go, but briefly:
   - It downloads the **"New Plant Diseases Dataset (Augmented)"** from Kaggle (you'll
     need a free Kaggle account and API key — the notebook tells you exactly where to
     get it).
   - It filters out just the 10 tomato classes.
   - It builds a MobileNetV2-based model and trains it.
   - It shows accuracy/loss graphs, a confusion matrix, and a Grad-CAM visualization.
   - At the end, it downloads two files to your computer: `tomato_model.h5` and
     `class_indices.json`.
5. **Keep these two downloaded files safe** — you need them for Part 2.

### Part 2 — Run the app locally first (to make sure everything works)

1. Create a new folder on your computer, e.g. `tomato-app/`.
2. Put these 4 files in it:
   - `app.py`
   - `requirements.txt`
   - `tomato_model_advanced.h5` (from Part 2 of the notebook — `app.py` loads this by default)
   - `class_indices.json` (from Part 1, unchanged in Part 2)
3. Get a free Gemini API key: go to
   [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey), sign in,
   click **Create API key**, copy it.
4. Open a terminal in that folder and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Your browser will open the app automatically (usually at `http://localhost:8501`).
   Paste your Gemini API key into the sidebar, upload a tomato leaf photo (you can
   grab a sample image from the dataset's `valid` folder, or just search "tomato leaf
   disease" online), and check that you get a prediction, a Grad-CAM heatmap, and an
   AI-generated report.

### Part 3 — Deploy it so anyone can use it via a link (optional but recommended)

1. Create a free GitHub account if you don't have one, and push your `tomato-app/`
   folder as a new repository (including the `.h5` and `.json` files — GitHub allows
   files up to 25MB via the web upload, or up to 100MB via git; the model file should
   be well under that).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   click **New app**, point it to your repository and `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy. You'll get a public link like `https://your-app-name.streamlit.app` that
   you can share with your sister's mentor or demo live.

