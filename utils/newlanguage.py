import urllib.request
from collections import defaultdict

import lxml.etree


def get_territory_languages():
    # Load the CLDR supplemental data XML
    url = "https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/supplementalData.xml"
    langxml = urllib.request.urlopen(url)
    langtree = lxml.etree.XML(langxml.read())

    territory_languages = {}

    # Iterate over each territory (country) in the XML
    for t in langtree.find("territoryInfo").findall("territory"):
        langs = {}

        # Iterate over languages spoken in the territory
        for l in t.findall("languagePopulation"):
            langs[l.get("type")] = {
                "percent": float(l.get("populationPercent")),
                "official": l.get("officialStatus")
                == "official",  # True if official language
            }

        # Add the languages and their info to the territory (country)
        territory_languages[t.get("type")] = langs

    return territory_languages


def map_languages_to_countries(territory_languages):
    # Initialize a dictionary where language codes map to a list of countries
    lang_to_countries = defaultdict(list)

    # Iterate over each country and its languages
    for country_code, languages in territory_languages.items():
        for lang_code, lang_info in languages.items():
            # Add country to the language if it's official or widely spoken
            lang_to_countries[lang_code].append(
                {
                    "country": country_code,
                    "percent": lang_info["percent"],
                    "official": lang_info["official"],
                }
            )

    return lang_to_countries


# Fetch territory languages
TERRITORY_LANGUAGES = get_territory_languages()

# Map languages to countries
LANGUAGE_TO_COUNTRIES = map_languages_to_countries(TERRITORY_LANGUAGES)

# Example: Get the countries where 'kn' (Kannada) is spoken
kannada_countries = LANGUAGE_TO_COUNTRIES.get("kn", [])
print(f"Countries where Kannada is spoken: {kannada_countries}")
