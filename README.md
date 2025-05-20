# Trash Material Classifier – ADA447 Midterm Project

**Author:** Selay Çetin  
**Course:** ADA447 – Introduction to Deep Learning  
**Term:** Spring 2025

---

##  Project Overview

This project classifies marine trash images by material type (e.g. plastic, metal, paper) using a deep learning model trained on the TrashCAN 1.0 dataset.

It was developed as part of the ADA447 midterm project and deployed using Gradio on Hugging Face Spaces.

---

##  Live Demo

👉 Try the app here:  
**[Hugging Face Space →](https://huggingface.co/spaces/slyctn/trash-classifier)**

---

##  Blog Post

Read the full project explanation and methodology:  
🔗 [Medium Blog Post →](<https://medium.com/@slyctn16/author-selay-%C3%A7etin-f611c09d8a4e>)

---

## Files Included

- `app.py` – Gradio web interface
- `requirements.txt` – Dependencies
- `trash_classifier.pkl` – ✅ Included in Hugging Face, **excluded here due to GitHub file size limits**

---

##  Notes

- The model was trained using FastAI `resnet34`
- Used COCO-style JSON annotations and DataBlock API
- All training, fine-tuning, evaluation, and deployment steps follow the course topics.docx requirements
