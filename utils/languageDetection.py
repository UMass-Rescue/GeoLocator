from langcodes import Language
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
#import easyocr
#from googletrans import Translator
#from utils.countryFromLanguages import get_location_from_language 
#from utils.locationDetailFromLanguageCode import get_location_from_languagecode


# List of 100 language codes
language_codes = [
    'en', 'fr', 'es', 'de', 'te', 'ja', 'kn', 'zh', 'it', 'pt', 'ru', 'ar', 'ko', 'nl', 'hi', 'sv', 'tr', 'pl',
    'uk', 'vi', 'el', 'th', 'he', 'cs', 'fi', 'hu', 'ro', 'da', 'id', 'no', 'sk', 'bg', 'hr', 'lt', 'sl', 'lv',
    'et', 'sr', 'ms', 'bs', 'fa', 'sq', 'ca', 'is', 'az', 'mk', 'mt', 'sw', 'af', 'ne', 'bn', 'ta', 'ml', 'ur',
    'mr', 'gu', 'pa', 'si', 'am', 'zu', 'yo', 'ha', 'ig', 'mg', 'or', 'my', 'km', 'lo', 'mn', 'hy', 'ka', 'tg',
    'uz', 'tk', 'kk', 'ky', 'tt', 'ps', 'sd', 'si', 'la', 'ga', 'cy', 'gd', 'eu', 'co', 'fy', 'lb', 'sm', 'to',
    'gl', 'mt', 'rn', 'so', 'ts', 've', 'xh', 'sn', 'ny', 'rw'
]

# Generate "Language text" for each language
texts = [f"{Language.get(lang_code).display_name()} text" for lang_code in language_codes]

# Print the generated list
#print(texts)



# Initialize the translator
translator = Translator()

# Function to translate a list of texts to English
def translate_to_english(texts, source_lang='auto'):
    translated_texts = []
    #print(texts)
    for text in texts:
        # Translate the text to English
        translated = translator.translate(text, src=source_lang, dest='en')
        #print(translated)
        translated_texts.append(translated.text)
    return translated_texts



# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image (replace with your image URL or local path)
#image_url = "/content/sample_data/w.jpg"
#image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
def get_lang_code(image_path):
    #print("OCR Rahasya Data")
    image = Image.open(image_path).convert("L")

    # Prepare image and prompt texts (you can create a list of languages for language detection)
    #texts = ["English text", "French text", "Spanish text", "German text","Telegu Text", "Japanese text","Kannada Text"]


    # Preprocess image and text
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # Compute logits per image and text
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Shape: [1, len(texts)]
        probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    # Find the most likely text (language)
    best_match_idx = torch.argmax(probs).item()
    probability_value = probs[0, best_match_idx].item()  # Get the probability as a float
    #print(int(np.max(probs)))
    detected_language = texts[best_match_idx]

    ###Added temporarily only for presenation
    if probability_value>0.3:
        return detected_language

    #print(image_path,": Detected Language:", detected_language, "with probability", probability_value)
    #lang_list = ['en']
    #if probability_value>0.3:
        #return ['en',Language.find(detected_language)]
        #lang_list.append(Language.find(detected_language).language)
    #text = pytesseract.image_to_string(image, lang='eng+fra')
    #print(lang_list)
    #print(text)
    #reader = easyocr.Reader(lang_list, gpu=False)
    # Step 1: Extract Text from the Image with EasyOCR
    #result = reader.readtext(image_path)
    #ocr_data = []
    #for (bbox, text, conf) in result:
        #print(f"Detected Text: '{text}' with confidence {conf:.2f}")
        #ocr_data.append(text)
    
    #print(ocr_data)
    #print("PyCountry",get_location_from_language(lang_list[-1]))
    #print("geopy",get_location_from_languagecode(lang_list[-1]))

    # if len(lang_list)>1:
    #     translated_texts = translate_to_english(ocr_data, source_lang=lang_list[-1])
    #     print("Translated Texts:")
    #     for text in translated_texts:
    #         print(text)



    
