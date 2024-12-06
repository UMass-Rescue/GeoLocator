import argparse
import json
import os
import time
import warnings

import torch

from geoclipModule import geoclip
from IndoorOutdoorClassifier.iodetector import run_iodetector
from utils import textExtraction

warnings.filterwarnings("ignore")
import shutil

from geoclip import GeoCLIP

from Evaluation import calc_accuracy

# Import functions from project modules
from geoclipModule.geoclip import detect_location_from_image
from IndoorOutdoorClassifier.iodetector import run_iodetector
from TextSpotter.Craft.textspot import run_craft
from utils.countryMappingFromLanguage import get_country_name, getcountry
from utils.languageDetection import get_lang_code
from utils.textExtraction import get_location_from_text
from utils.translator import translate


def append_to_json(file_path, data):
    """Appends data to a JSON file, creating it if it doesn't exist."""

    if os.path.exists(file_path):
        with open(file_path, "r+") as file:
            # Load existing data
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  # File is empty or invalid JSON

            # Append new data
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]

            # Write updated data back to file
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        # Create a new file
        with open(file_path, "w") as file:
            json.dump([data], file, indent=4)


def start(directory, gt, outputJson):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]  # Add more if needed
    image_files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            if any(filename.endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, filename))

    print(f"There are {len(image_files)} images in the folder selected to be processed")

    results = []  # Store results for each processed image
    temp_folder = "temp/"  # Temporary folder to store processed images
    model = GeoCLIP()
    output_path = outputJson
    os.remove(output_path) if os.path.exists(output_path) else None
    for img_file in image_files:
        shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(
            temp_folder, exist_ok=True
        )  # Create temp folder if it doesn't exist
        print("Processing image:", img_file)

        # Run Indoor/Outdoor detector
        print("Predicting Indoor Outdor and Scene Type")
        io_result = run_iodetector(img_file)
        print("IO Detection Result:", io_result)

        # Run GeoClip model for location detection based on the image
        print("Running GEOCLIP")
        geo_result = detect_location_from_image(model, img_file, io_result)
        print("Geo Detection Result:", geo_result)
        textspot_results = []  # Store OCR results for the current image
        try:
            # Run the CRAFT model for text detection on the image
            run_craft(
                image_path=img_file,
                result_folder=temp_folder,
                trained_model="TextSpotter/Craft/weights/craft_mlt_25k.pth",
                text_threshold=0.7,  # Lower threshold for quicker detection
                low_text=0.3,  # Higher value to exclude faint text
                link_threshold=0.4,  # Higher value for faster linkage
                cuda=False,  # Enable GPU acceleration for faster processing
                canvas_size=1280,  # Reduce canvas size for faster processing
                mag_ratio=1.5,  # Lower magnification for faster resizing
                poly=False,  # Skip polygonal representation for simpler processing
                refine=False,  # Keep refinement disabled
            )
        except Exception as e:
            print(f"Error running CRAFT on {img_file}: {e}")
            continue  # Skip to next image if there's an error
        if len(os.listdir(temp_folder)) > 2:
            language_detected = set()
            countries_detected = set()
            languages, locations = (
                [],
                [],
            )  # Initialize lists for languages and locations detected from text
            for processed_img in os.listdir(temp_folder):
                if processed_img.endswith((".jpg", ".png")):
                    processed_path = os.path.join(temp_folder, processed_img)

                    # Detect language and location from the text in the processed image
                    language = get_lang_code(processed_path)
                    if language:
                        language_detected.add(language.split(" ")[0])
                        country = getcountry(language.split(" ")[0])
                        for c in country:
                            countries_detected.add(get_country_name(c["country"]))

                    language, location = get_location_from_text(processed_path)
                    languages.append(language)
                    locations.extend(location)

            # Append text detection results to the results list
            textspot_results.append(
                {
                    "Languages Detected": list(set(languages)),
                    "Locations Detected from Text": list(set(locations)),
                }
            )
            language_detected.add("English")

            # Merge textspot results into the geo_result
            geo_result["Locations Detected from Text"] = list(set(locations))
            geo_result["Languages Detected Method 2"] = list(language_detected)
            geo_result["Countries Possible from Languages Spotted"] = list(
                countries_detected
            )

            results.append(geo_result)

        # Clean up the temporary folder after processing
        else:
            print("No Text Spotted in the image")
            results.append(geo_result)

    results = translate(results)
    calc_accuracy.top1(results, gt)

    # Define the output JSON file path from inputs and write results to it
    try:
        append_to_json(output_path, results)
        print("Results written to:", output_path)
    except Exception as e:
        print(f"Failed to write results: {e}")


def main():
    print(torch.cuda.is_available())
    image_folder = "Evaluation/Images"
    GroundTruth = "Evaluation/Labels.csv"
    output_csv = "Evaluation/op.csv"
    start(image_folder, GroundTruth, output_csv)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))