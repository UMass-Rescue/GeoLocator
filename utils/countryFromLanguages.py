import pycountry


# Function to get the location from a language code
def get_location_from_language(language_code):
    print(pycountry.countries)
    try:
        # Get the language information
        language = pycountry.languages.get(alpha_2=language_code)
        print(language)
        if not language:
            language = pycountry.languages.get(alpha_3=language_code)

        if language:
            print("hi")
            # Attempt to find the country for the given language
            countries = [
                country.name
                for country in pycountry.countries
                if language.name in country.official_languages
                or language.name in getattr(country, "languages", [])
            ]
            print(countries, "countriess")
            if countries:
                print(countries)
                return countries
            else:
                return (
                    f"No specific country found for the language code '{language_code}'"
                )
        else:
            return f"Invalid language code '{language_code}'"
    except Exception as e:
        return str(e)
