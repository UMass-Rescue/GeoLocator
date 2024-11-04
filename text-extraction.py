# Import necessary libraries for OCR and NER
import pytesseract
import spacy
from PIL import Image
import cv2
import re

# Step 1: OCR
image_path = 'image/welcome-to-ny.jpg'
image = cv2.imread(image_path)

# Preprocess the image for OCR
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ocr_result = pytesseract.image_to_string(gray_image)
print("Extracted Text:")
print(ocr_result)

# Clean the OCR result
def clean_text(text):
    clean_text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    clean_text = re.sub(r'\n', ' ', clean_text)  # Replace line breaks with space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Remove extra spaces
    return clean_text

cleaned_text = clean_text(ocr_result)
print("Cleaned Text:", cleaned_text)

# Step 2: NER (Named Entity Recognition)
nlp = spacy.load('en_core_web_trf')
doc = nlp(cleaned_text)
locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
print("Recognized Locations (NER):", locations)
