import numpy as np
import torch
from langcodes import Language
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# List of 100 language codes
language_codes = [
    "en",
    "fr",
    "es",
    "de",
    "te",
    "ja",
    "kn",
    "zh",
    "it",
    "pt",
    "ru",
    "ar",
    "ko",
    "nl",
    "hi",
    "sv",
    "tr",
    "pl",
    "uk",
    "vi",
    "el",
    "th",
    "he",
    "cs",
    "fi",
    "hu",
    "ro",
    "da",
    "id",
    "no",
    "sk",
    "bg",
    "hr",
    "lt",
    "sl",
    "lv",
    "et",
    "sr",
    "ms",
    "bs",
    "fa",
    "sq",
    "ca",
    "is",
    "az",
    "mk",
    "mt",
    "sw",
    "af",
    "ne",
    "bn",
    "ta",
    "ml",
    "ur",
    "mr",
    "gu",
    "pa",
    "si",
    "am",
    "zu",
    "yo",
    "ha",
    "ig",
    "mg",
    "or",
    "my",
    "km",
    "lo",
    "mn",
    "hy",
    "ka",
    "tg",
    "uz",
    "tk",
    "kk",
    "ky",
    "tt",
    "ps",
    "sd",
    "si",
    "la",
    "ga",
    "cy",
    "gd",
    "eu",
    "co",
    "fy",
    "lb",
    "sm",
    "to",
    "gl",
    "mt",
    "rn",
    "so",
    "ts",
    "ve",
    "xh",
    "sn",
    "ny",
    "rw",
]

# Generate "Language text" for each language
texts = [
    f"{Language.get(lang_code).display_name()} text" for lang_code in language_codes
]


# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_lang_code(image_path):
    image = Image.open(image_path).convert("L")

    # Preprocess image and text
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # Compute logits per image and text
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Shape: [1, len(texts)]
        probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    # Find the most likely text (language)
    best_match_idx = torch.argmax(probs).item()
    probability_value = probs[
        0, best_match_idx
    ].item()  # Get the probability as a float
    # print(int(np.max(probs)))
    detected_language = texts[best_match_idx]

    # Added temporarily only for presenation
    if probability_value > 0.3:
        return detected_language
