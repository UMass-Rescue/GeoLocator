# Import necessary libraries
import os
import pandas as pd
import re
from tqdm import tqdm
from PIL import Image
import easyocr
import spacy
from langdetect import detect
from spacy.matcher import PhraseMatcher

# # List of image paths
image_paths = [
    'testImages/ffbaa76bd8d172cf.jpg',
    'testImages/ffda1919cf90a8eb.jpg',
    'testImages/ffdf2c012fcee84d.jpg',
    'testImages/ffe3f4c718e9ad0d.jpg',
    'testImages/fffb31ec87802a5a.jpg'
]

# Initialize EasyOCR reader with multilingual support (modify languages as needed)


# Define functions for text cleaning, language detection, and NER post-processing
def clean_text(text):
    """Remove special characters, short words, and extra spaces."""
    clean_text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    clean_text = re.sub(r'\b\w{1,2}\b', '', clean_text)  # Remove short words
    clean_text = re.sub(r'\n', ' ', clean_text)  # Replace line breaks with space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Remove extra spaces
    return clean_text

def post_process_ner(entities):
    """Remove trailing characters and filter out empty strings."""
    processed = [re.sub(r'\bi\b$', '', ent).strip() for ent in entities]
    return list(filter(lambda x: x, processed))

# Initialize PhraseMatcher for known locations
def initialize_phrase_matcher(nlp):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    known_locations = [
        "San Jose", "California", "New York", "Tokyo", "Seoul", "Paris",
        "London", "Rome", "Munich", "Beijing", "Shanghai",
        "دبي", "القاهرة", "Riyadh", "서울", "東京", "मुंबई",
        "প্যারিস", "فلورنسا"
    ]
    patterns = [nlp.make_doc(loc) for loc in known_locations]
    matcher.add("LocationMatcher", patterns)
    return matcher

def get_location_from_text(image):
    #print(f"\nProcessing image")
    reader = easyocr.Reader(['en'], gpu=True)
    # Step 1: Extract Text from the Image with EasyOCR
    result = reader.readtext(image)
    ocr_data = []
    for (bbox, text, conf) in result:
        #print(f"Detected Text: '{text}' with confidence {conf:.2f}")
        ocr_data.append(text)

    # Concatenate all detected text for further processing
    full_text = ' '.join(ocr_data)
    #print("\nFull Extracted Text:")
    #print(full_text)

    # Step 2: Detect Language of Extracted Text
    #print("\nDetecting language of the extracted text...")
    try:
        detected_lang = detect(full_text)
        #print(f"Detected Language: {detected_lang}")
    except Exception as e:
        #print(f"Language detection failed: {e}")
        detected_lang = 'en'

    # Step 3: Load spaCy NER Model Based on Language Detection
    if detected_lang == 'en':
        try:
            nlp = spacy.load('en_core_web_trf')
        except OSError:
            #print("Downloading 'en_core_web_trf' model...")
            os.system('python -m spacy download en_core_web_trf')
            nlp = spacy.load('en_core_web_trf')
    else:
        try:
            nlp = spacy.load('xx_ent_wiki_sm')
        except OSError:
            #print("Downloading 'xx_ent_wiki_sm' model...")
            os.system('python -m spacy download xx_ent_wiki_sm')
            nlp = spacy.load('xx_ent_wiki_sm')

    # Initialize the PhraseMatcher with known locations
    matcher = initialize_phrase_matcher(nlp)

    # Step 4: Clean the OCR Text
    cleaned_text = clean_text(full_text)
    #print("\nCleaned Text:")
    #print(cleaned_text)

    # Step 5: Perform Named Entity Recognition (NER) to Find Locations
    doc = nlp(cleaned_text)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  # GPE = Geopolitical Entity
    locations = post_process_ner(locations)
    #print("\nRecognized Locations (NER):", locations)

    # Step 6: Use PhraseMatcher to Find Known Locations
    def apply_phrase_matcher(text):
        doc = nlp(text)
        matches = matcher(doc)
        matched_locs = [doc[start:end].text for _, start, end in matches]
        return matched_locs

    matched_locations = apply_phrase_matcher(cleaned_text)
    #print("\nMatched Locations (PhraseMatcher):", matched_locations)

    # Combine NER and PhraseMatcher results
    all_locations = set(locations + matched_locations)
    #print("\nCombined Recognized Locations (NER + PhraseMatcher):", list(all_locations))

    # Step 7: Regex Fallback for Additional Matching
    def regex_fallback(text):
        """Fallback method using regex to match known patterns of locations."""
        pattern = r'\b(San Jose|California|New York|Tokyo|Seoul|Paris|London|Rome|Munich|Beijing|Shanghai|دبي|القاهرة|Riyadh|서울|東京|मुंबई|প্যারিস|فلورنسا)\b'
        return re.findall(pattern, text, re.IGNORECASE)

    # Use regex fallback if no locations are found
    if not all_locations:
        #print("\nFallback: Using regex matching.")
        all_locations = regex_fallback(cleaned_text)

    #print("\nFinal Recognized Locations:", list(all_locations))
    return detected_lang,all_locations

