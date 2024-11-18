# Import necessary modules
import json
import os
import shutil
from typing import List, TypedDict
import warnings
warnings.filterwarnings('ignore') 


# Import Flask-ML and other components for creating an ML server
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (BatchFileInput, BatchTextResponse,
                                             FileInput, FileResponse, FileType,
                                             InputSchema, InputType,
                                             NewFileInputType, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor)

# Import functions from project modules
from geoclipModule.geoclip import detect_location_from_image
from IndoorOutdoorClassifier.iodetector import run_iodetector
from TextSpotter.Craft.textspot import run_craft
from utils.textExtraction import get_location_from_text

# Initialize Flask-ML server
server = MLServer(__name__)

# Add application metadata
server.add_app_metadata(
    name="GeoLocator",
    author="Islam",
    version="0.3.4",
    info=load_file_as_string("./README.md"),
)


# Define schema for image processing task inputs and outputs
def image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="image_input", label="Upload Images", input_type=InputType.BATCHFILE
            ),
            InputSchema(
                key="output_path",
                label="Output JSON Path",
                input_type=NewFileInputType(
                    default_name="output.json",
                    default_extension=".json",
                    allowed_extensions=[".json"],
                ),
            ),
        ],
        parameters=[],
    )


# Define types for the image inputs and parameters
class ImageInputs(TypedDict):
    image_input: BatchFileInput
    output_path: FileInput


class ImageParameters(TypedDict): ...


# Function to append data to an existing JSON file, or create it if it doesn't exist
def append_to_json(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, "r+") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
            # Append new data to existing data if it's a list
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, "w") as file:
            json.dump([data], file, indent=4)


# Define route for processing images
@server.route(
    "/process_images", task_schema_func=image_task_schema, short_title="Result"
)
def process_images(inputs: ImageInputs, parameters: ImageParameters) -> ResponseBody:
    results = []  # Store results for each processed image
    temp_folder = "temp/"
    shutil.rmtree(temp_folder, ignore_errors=True)
    output_path = inputs["output_path"].path
    os.remove(output_path) if os.path.exists(output_path) else None


    os.makedirs(temp_folder, exist_ok=True)

    # Process each image uploaded by the user
    for img_file in inputs["image_input"].files:
        print("Processing image:", img_file.path)

        # Run Indoor/Outdoor detector
        print("Predicting Indoor Outdor and Scene Type")
        io_result = run_iodetector(img_file.path)
        print("IO Detection Result:", io_result)

        # Run GeoClip model for location detection based on the image
        print("Prediction Location using GeoCLIP")
        geo_result = detect_location_from_image(img_file.path, io_result)
        print("Geo Detection Result:", geo_result)

        print("Checking for text")
        textspot_results = []
        try:
            # Run the CRAFT model for text detection on the image
            run_craft(
                image_path=img_file.path,
                result_folder=temp_folder,
                trained_model="TextSpotter/Craft/weights/craft_mlt_25k.pth",
                text_threshold=0.7,
                low_text=0.3,
                link_threshold=0.4,
                cuda=False,
                canvas_size=1280,
                mag_ratio=1.5,
                poly=False,
                refine=False,
            )
        except Exception as e:
            print(f"Error running CRAFT on {img_file.path}: {e}")
            continue

        languages, locations = (
            [],
            [],
        )  # Initialize lists for languages and locations detected from text
        for processed_img in os.listdir(temp_folder):
            if processed_img.endswith((".jpg", ".png")):
                processed_path = os.path.join(temp_folder, processed_img)
                try:
                    # Detect language and location from the text in the processed image
                    language, location = get_location_from_text(processed_path)
                    languages.append(language)
                    locations.extend(location)
                except Exception as e:
                    print(f"Error processing {processed_path}: {e}")

        # Append text detection results to the results list
        textspot_results.append(
            {
                "Languages Detected": list(set(languages)),
                "Locations Detected from Text": list(set(locations)),
            }
        )

        # Merge textspot results into the geo_result
        geo_result["Languages Detected"] = list(set(languages))
        geo_result["Locations Detected from Text"] = list(set(locations))
        results.append(geo_result)

    # Define the output JSON file path from inputs and write results to it
    try:
        append_to_json(output_path, results)
        print("Results written to:", output_path)
    except Exception as e:
        print(f"Failed to write results: {e}")

    # Clean up the temporary folder after processing
    shutil.rmtree(temp_folder)

    # Return the output JSON file to the user as a download
    return ResponseBody(
        root=FileResponse(
            file_type=FileType.JSON,
            path=output_path,
            title="Report",
        )
    )


# Run the server if this script is executed directly
if __name__ == "__main__":
    server.run()
