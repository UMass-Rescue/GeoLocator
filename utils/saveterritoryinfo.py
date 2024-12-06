import json
import urllib.request

import lxml.etree


def get_territory_languages():
    # Fetch the CLDR XML data from the URL
    url = "https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/supplementalData.xml"
    langxml = urllib.request.urlopen(url)
    langtree = lxml.etree.XML(langxml.read())

    territory_languages = {}

    # Parse territory (country) and language data
    for t in langtree.find("territoryInfo").findall("territory"):
        langs = {}
        for l in t.findall("languagePopulation"):
            langs[l.get("type")] = {
                "percent": float(l.get("populationPercent")),
                "official": l.get("officialStatus") == "official",
            }
        territory_languages[t.get("type")] = langs

    return territory_languages


# Fetch the territory languages
territory_languages = get_territory_languages()

# Save the territory languages dictionary to a JSON file
with open("territory_languages.json", "w") as f:
    json.dump(territory_languages, f, indent=4)
