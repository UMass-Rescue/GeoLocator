import json
from collections import defaultdict

import pycountry


# Function to get the ISO 639-1 code from a language name
def get_language_code(language_name):
    try:
        # Use pycountry to lookup the language by name
        language = pycountry.languages.lookup(language_name)
        return language.alpha_2  # ISO 639-1 code (2-letter code)
    except LookupError:
        return None


def map_languages_to_countries(territory_languages):
    lang_to_countries = defaultdict(list)

    # Map each language to the countries where it is spoken
    for country_code, languages in territory_languages.items():
        for lang_code, lang_info in languages.items():
            lang_to_countries[lang_code].append(
                {
                    "country": country_code,
                    "percent": lang_info["percent"],
                    "official": lang_info["official"],
                }
            )

    return lang_to_countries


def getcountry(language):
    # Load the saved territory languages data
    with open("territory_languages.json", "r") as f:
        territory_languages = json.load(f)

    # Create the language to countries mapping
    language_to_countries = map_languages_to_countries(territory_languages)

    code = get_language_code(language)
    # print(code)
    countries = language_to_countries.get(code, [])
    return countries


def get_country_name(country_code):
    try:
        # Lookup the country by its ISO 3166-1 alpha-2 code (2-letter code)
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name if country else None
    except KeyError:
        return None
