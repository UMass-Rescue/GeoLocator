# Import necessary libraries
import os
import re

import easyocr
import spacy
from langdetect import detect
from spacy.matcher import PhraseMatcher


# Function to clean text
def clean_text(text):
    """Remove special characters, short words, and extra spaces."""
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\b\w{1,2}\b", "", text)  # Remove short words
    text = re.sub(r"\n", " ", text)  # Replace line breaks with spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


# Function to post-process NER results
def post_process_ner(entities):
    """Remove trailing characters and filter out empty strings."""
    processed = [re.sub(r"\bi\b$", "", ent).strip() for ent in entities]
    return list(filter(None, processed))


# Initialize PhraseMatcher with known locations
def initialize_phrase_matcher(nlp):
    """Initialize a PhraseMatcher with predefined location patterns."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    known_locations = [
        "San Jose",
        "California",
        "New York",
        "Tokyo",
        "Seoul",
        "Paris",
        "London",
        "Rome",
        "Munich",
        "Beijing",
        "Shanghai",
        "دبي",
        "القاهرة",
        "Riyadh",
        "서울",
        "東京",
        "मुंबई",
        "প্যারিস",
        "فلورنسا",
    ]
    patterns = [nlp.make_doc(location) for location in known_locations]
    matcher.add("LocationMatcher", patterns)
    return matcher


# Function to extract text and detect locations from an image
def get_location_from_text(image):
    """Process an image to extract text, detect language, and find locations."""
    reader = easyocr.Reader(["en"], gpu=True)  # Enable GPU for faster processing

    # Step 1: Extract text using EasyOCR
    result = reader.readtext(image)
    extracted_text = " ".join([text for (_, text, _) in result])

    # Step 2: Detect language
    try:
        detected_lang = detect(extracted_text)
    except Exception:
        detected_lang = "en"  # Default to English if detection fails

    # Step 3: Load appropriate spaCy model based on language
    model_name = "en_core_web_trf" if detected_lang == "en" else "xx_ent_wiki_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name)

    # Initialize PhraseMatcher
    matcher = initialize_phrase_matcher(nlp)

    # Step 4: Clean extracted text
    cleaned_text = clean_text(extracted_text)

    # Step 5: Perform Named Entity Recognition (NER) to find geopolitical entities
    doc = nlp(cleaned_text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    locations = post_process_ner(locations)

    # Step 6: Use PhraseMatcher to find known locations
    def apply_phrase_matcher(text):
        doc = nlp(text)
        matches = matcher(doc)
        return [doc[start:end].text for _, start, end in matches]

    matched_locations = apply_phrase_matcher(cleaned_text)

    # Combine NER and PhraseMatcher results
    all_locations = set(locations + matched_locations)

    # Step 7: Fallback regex matching for additional location detection
    if not all_locations:
        pattern = r"\b(San Jose|California|New York|Tokyo|Seoul|Paris|London|Rome|Munich|Beijing|Shanghai|دبي|القاهرة|Riyadh|서울|東京|मुंबई|প্যারিস|فلورنسا)\b"
        all_locations = re.findall(pattern, cleaned_text, re.IGNORECASE)

    return detected_lang, list(all_locations)
