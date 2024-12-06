import json

from deep_translator import GoogleTranslator


def translate_text(text, translator):
    """Translates text to English if not already in English."""
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def translate_json(data, translator):
    """Recursively translates non-English fields in the JSON data."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                data[key] = translate_json(value, translator)
            elif isinstance(value, str):
                data[key] = translate_text(value, translator)
    elif isinstance(data, list):
        return [
            (
                translate_json(item, translator)
                if isinstance(item, (dict, list))
                else translate_text(item, translator)
            )
            for item in data
        ]
    return data


def translate(data):
    # Initialize the translator to English
    translator = GoogleTranslator(source="auto", target="en")

    # Translate the JSON data
    translated_data = translate_json(data, translator)

    return translated_data
