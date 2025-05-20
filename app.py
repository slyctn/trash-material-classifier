import gradio as gr
from fastai.vision.all import *

# Load the exported model
learn = load_learner('trash_classifier.pkl')
example_images = [
    ["vid_000161_frame0000015.jpg"],
    ["vid_000246_frame0000032.jpg"],
    ["vid_000253_frame0000003.jpg"],
    ["vid_000299_frame0000011.jpg"],
    ["vid_000555_frame0000028.jpg"],
    ["vid_000547_frame0000020.jpg"]
    
]

def classify_image(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return f"Prediction: {pred}\nConfidence: {probs[idx]:.2%}"

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Trash Material Classifier",
    description="Upload a marine trash image to classify its material type (plastic, metal, paper, etc.)",
    examples=example_images
).launch()
