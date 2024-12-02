from googletrans import Translator

translator = Translator()


def auto_translate(data, dest_lang="en"):
    if isinstance(data, dict):
        return {key: auto_translate(value, dest_lang) for key, value in data.items()}
    elif isinstance(data, list):
        return [auto_translate(item, dest_lang) for item in data]
    elif isinstance(data, str):
        # Auto-detect source language and translate
        translated = translator.translate(data, dest=dest_lang)
        return translated.text
    else:
        return data

# Translate JSON data
#translated_data = auto_translate(data)