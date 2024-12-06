import pycountry
from geopy.geocoders import Nominatim
from langcodes import Language

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")


# Function to get location details from a language code
def get_location_from_languagecode(language_code):
    # Get the language object
    language_name = pycountry.languages.get(alpha_2=language_code).name
    print(language_name)
    location = geolocator.geocode(f"Primary region where {language_name} is spoken")
    if location:
        return location.address

    language = Language.get(language_code)
    print(language)

    # Extract the likely country name
    country_name = language.describe().get("territory")
    if not country_name:
        return "Could not determine country from the language."

    # Geocode to find country and state details
    location = geolocator.geocode(country_name)
    if location:
        return {
            "country": location.address.split(",")[-1].strip(),
            "state": location.address.split(",")[-2].strip(),
        }
    else:
        return "Location details not found."
