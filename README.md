# Text Extraction with OCR and NER
This project performs Optical Character Recognition (OCR) and Named Entity Recognition (NER) on images using EasyOCR and spaCy. We utilize the TextOCR dataset to extract text from images, detect the language, and identify geopolitical entities (locations) within the extracted text.

## Key Dependencies

- **easyocr**: For OCR text extraction from images.
- **spacy**: For Named Entity Recognition (NER) with language models `en_core_web_trf` (English transformer-based model) and `xx_ent_wiki_sm` (multilingual).
- **langdetect**: For detecting the language of the extracted text.
- **tqdm**: For displaying progress bars.
- **pandas**: For data handling and analysis.
- **Pillow**: For image processing.

## Installation

### Clone the repository:
```bash
git clone https://github.com/UMass-Rescue/GeoLocator.git
cd GeoLocator
```
### Install the requirements:
```bash
pip install -r requirements.txt
```
### Download the required spaCy models:
```bash
python -m spacy download en_core_web_trf
python -m spacy download xx_ent_wiki_sm
```
# Usage
Set Up Image Paths: Update the `image_paths` list in the script with the paths to your images.

Run the Script:
```bash
python text-extraction.py
```
The script will:

- Extract text from each image using EasyOCR.
- Detect the language of the extracted text.
- Load the appropriate spaCy NER model based on the detected language.
- Clean the OCR text for better NER performance.
- Use spaCy NER and PhraseMatcher to detect locations.
- Use a regex fallback to match additional locations if needed.

# Code Structure
- OCR Extraction: Uses EasyOCR to extract text from images.
- Language Detection: Detects the language of the extracted text using `langdetect` to choose the appropriate NER model.
- Text Cleaning: Cleans the OCR output text to improve NER performance.
- Named Entity Recognition (NER): Identifies geopolitical entities (locations) using spaCy.
- PhraseMatcher: Matches known locations using spaCy's PhraseMatcher.
- Regex Fallback: Uses regex to match additional locations based on predefined patterns if no locations are detected.

# Notes
- Ensure you have a GPU available to leverage `gpu=True` in EasyOCR for faster processing.
- Update the `known_locations` list in the PhraseMatcher function to add or modify location patterns as needed.
- Modify the `image_paths` list with paths to the images you want to process.

