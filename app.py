import gradio as gr
from fastai.vision.all import *

# Load the exported model
learn = load_learner('trash_classifier.pkl')

def classify_image(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return f"Prediction: {pred}\nConfidence: {probs[idx]:.2%}"

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Trash Material Classifier",
    description="Upload a marine trash image to classify its material type (plastic, metal, paper, etc.)"
).launch()
