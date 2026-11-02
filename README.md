# Intel Image Classification Web App

A Streamlit web application that classifies photos of natural and urban scenes into six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The model is an EfficientNetB0 base (ImageNet weights, transfer learning) with a custom
classification head, trained in Google Colab and fine-tuned for this dataset.

## Results

- **Test accuracy:** 90% (sealed test set, 3,000 images)
- **Validation accuracy:** 91.2% (after fine-tuning)

## Dataset

The [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
dataset (~25k images at 150x150, 6 classes).

## How to run locally

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Make sure `intel_weights_fp16.npz` is in the same folder as `app.py`, then run:

```
streamlit run app.py
```

## Project structure

- `app.py` — the Streamlit web app
- `intel_weights_fp16.npz` — the trained weights in float16 (~7 MB; small enough for
  GitHub's 25 MB web-upload limit, with no meaningful accuracy loss)
- `requirements.txt` — Python dependencies
