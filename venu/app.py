from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from googletrans import Translator
import cv2
import os
import csv
from difflib import SequenceMatcher

app = Flask(__name__)

# Load ground truth data from CSV file
def load_ground_truth(csv_file):
    ground_truth_data = {}
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            ground_truth_data[row['imagename']] = row['image_text']
    return ground_truth_data

# Set the path to your ground truth CSV file
ground_truth_file = r'F:\ocr\venu\data.csv'
ground_truth_data = load_ground_truth(ground_truth_file)

# Specify the path to the Tesseract executable
# Update this path based on your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def calculate_precision(predicted_text, ground_truth):
    # Calculate true positive (TP), false positive (FP), and false negative (FN)
    common_chars = set(predicted_text) & set(ground_truth)
    true_positives = len(common_chars)
    false_positives = len(set(predicted_text) - common_chars)
    false_negatives = len(set(ground_truth) - common_chars)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return precision
def ensure_upload_directory():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    upload_dir = os.path.join(script_dir, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
def calculate_recall(predicted_text, ground_truth):
    common_chars = set(predicted_text) & set(ground_truth)
    true_positives = len(common_chars)
    false_negatives = len(set(ground_truth) - common_chars)

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return recall
def calculate_confusion_matrix(predicted_text, ground_truth):
    common_chars = set(predicted_text) & set(ground_truth)
    
    true_positives = len(common_chars)
    false_positives = len(set(predicted_text) - common_chars)
    false_negatives = len(set(ground_truth) - common_chars)
    
    true_negatives = len(set(ground_truth + predicted_text) == 0)

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }
def calculate_f1_score(precision, recall):
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score
@app.route('/', methods=['GET', 'POST'])
def perform_ocr():
    text = ""
    selected_language = "en"  # Default language is English

    if request.method == 'POST' and 'file' in request.files:
        ensure_upload_directory()

        uploaded_file = request.files['file']

        # Save the uploaded file using the full file path
        file_path = os.path.abspath(os.path.join('uploads', uploaded_file.filename))
        uploaded_file.save(file_path)

        # Update ground truth data for the current file
        ground_truth_text = ground_truth_data.get(os.path.basename(file_path), '')
        ground_truth_data[file_path] = ground_truth_text
        selected_language = request.form.get('language', 'en')

        try:
            translated_text = perform_tesseract_ocr(file_path, selected_language, ground_truth_text)
            accuracy = calculate_accuracy(translated_text, ground_truth_text)
            precision = calculate_precision(translated_text, ground_truth_text)
            recall = calculate_recall(translated_text, ground_truth_text)
            f1_score = calculate_f1_score(precision, recall)
          
           
            return {'translated_text': translated_text, 'accuracy': accuracy,'precision': precision,'recall': recall,'f1_score': f1_score}

        except Exception as e:
            text = f"OCR process failed: {str(e)}"

        return jsonify({'text': text, 'selected_language': selected_language,'precision': None, 'recall': None,'f1_score': None,'confusion_matrix': None})

    return render_template('index.html', text=text, selected_language=selected_language)

def calculate_accuracy(predicted_text, ground_truth):
    if ground_truth:
        # Use SequenceMatcher to calculate similarity ratio
        similarity_ratio = SequenceMatcher(None, predicted_text, ground_truth).ratio()
        accuracy = round(similarity_ratio * 100, 2)
        return accuracy
    else:
        return None

def perform_tesseract_ocr(file_path, target_language='en', ground_truth=''):
    try:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = cv2.imread(file_path)
            
            # Preprocess the image using OpenCV if needed
            # For example, you can convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Perform OCR on the preprocessed image
            text = pytesseract.image_to_string(gray_img)
        elif file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
        else:
            supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pdf']
            text = f'Unsupported file format. Supported formats: {", ".join(supported_formats)}'
    except Exception as e:
        text = f"OCR process failed: {str(e)}"

    # Translate the text
    translator = Translator()
    extracted_text = text
    translated_text = translator.translate(extracted_text, dest=target_language).text

    return translated_text

if __name__ == '__main__':
    ensure_upload_directory()
    app.run(debug=True)