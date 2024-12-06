from babel import Locale


def get_country_from_language(language_code):
    try:
        locale = Locale.parse(language_code)

        # Print the language name in English and its primary territory
        language_name = locale.get_display_name("en")  # Display the language in English
        territories = locale.territories  # Dictionary of territories for the language

        print(f"Language: {language_name}")
        print("Territories:", territories)
    except ValueError as e:
        print("Error parsing language code:", e)
        return "Country details not found."


# Example usage
get_country_from_language("kn")  # Kannada
