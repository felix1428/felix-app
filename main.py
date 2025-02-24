from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from fpdf import FPDF
import uuid
from datetime import datetime

app = Flask(__name__)

# Load trained models
mri_model = load_model('models/brain.h5')
kidney_model = load_model('models/kidney.h5')

# Class labels
mri_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
kidney_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Define folders
UPLOAD_FOLDER = './uploads'
REPORTS_FOLDER = './reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# MRI Tumor Prediction Function
def predict_mri(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = mri_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)

    return mri_labels[predicted_class_index], confidence_score

# Kidney Disease Prediction Function
def predict_kidney(image_path):
    img = load_img(image_path, target_size=(28, 28))  # Resize to match model input size
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = kidney_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)

    return kidney_labels[predicted_class_index], confidence_score

def get_suggestion(scan_type, predicted_class):
    suggestions = {
        "mri": {
            "pituitary": "Consult a neurologist for further evaluation.",
            "glioma": "Seek immediate medical attention from an oncologist.",
            "notumor": "No tumor detected. Maintain regular check-ups.",
            "meningioma": "Consult a neurosurgeon for potential treatment options."
        },
        "kidney": {
            "Cyst": "Typically benign, but follow up with a urologist is recommended.",
            "Normal": "No abnormalities detected. Continue with routine health check-ups.",
            "Stone": "Increase water intake and consult a doctor for potential treatment.",
            "Tumor": "Immediate consultation with an oncologist is advised."
        }
    }
    return suggestions.get(scan_type, {}).get(predicted_class, "No suggestion available.")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # Your about page template

@app.route('/brain', methods=['GET', 'POST'])
def brain():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('brain.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('brain.html', error="No file selected")
        
        if file:
            # Save file with a unique name
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Predict
            tumor_class, confidence = predict_mri(file_path)
            suggestion = get_suggestion("mri", tumor_class)
            
            # Generate PDF
            pdf_filename = f"{os.path.splitext(unique_filename)[0]}.pdf"
            report_text = f"Prediction: {tumor_class}\nConfidence: {confidence*100:.2f}%\n\nSuggestion:\n{suggestion}"
            pdf_path = generate_pdf(report_text, pdf_filename)
            
            return render_template('brain.html',
                                  result=tumor_class,
                                  confidence=f"{confidence*100:.2f}%",
                                  suggestion=suggestion,
                                  file_path=f"uploads/{unique_filename}",
                                  pdf_path=f"download_report/{pdf_filename}")
    
    return render_template('brain.html')

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('kidney.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('kidney.html', error="No file selected")
        
        if file:
            # Save file with a unique name
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Predict
            kidney_class, confidence = predict_kidney(file_path)
            suggestion = get_suggestion("kidney", kidney_class)
            
            # Generate PDF
            pdf_filename = f"{os.path.splitext(unique_filename)[0]}.pdf"
            report_text = f"Prediction: {kidney_class}\nConfidence: {confidence*100:.2f}%\n\nSuggestion:\n{suggestion}"
            pdf_path = generate_pdf(report_text, pdf_filename)
            
            return render_template('kidney.html',
                                  result=kidney_class,
                                  confidence=f"{confidence*100:.2f}%",
                                  suggestion=suggestion,
                                  file_path=f"uploads/{unique_filename}",
                                  pdf_path=f"download_report/{pdf_filename}")
    
    return render_template('kidney.html')

def generate_pdf(report_text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report_text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
    pdf.output(pdf_path)
    return pdf_path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORTS_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)  