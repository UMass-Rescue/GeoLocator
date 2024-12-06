from countryFromLanguages import get_location_from_language
from locationDetailFromLanguageCode import get_location_from_languagecode

print("PyCountry", get_location_from_language("kn"))
print("geopy", get_location_from_languagecode("kn"))
