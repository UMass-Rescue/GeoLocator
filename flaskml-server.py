from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    BatchFileInput, BatchTextResponse, ResponseBody, TaskSchema, InputSchema, InputType, ParameterSchema, TextParameterDescriptor
)
from geoclipModule.geoclip import detect_location_from_image
from IndoorOutdoorClassifier.iodetector import run_iodetector
from TextSpotter.Craft.textspot import run_craft
from utils.textExtraction import get_location_from_text
import os
import json
import shutil
from typing import TypedDict, List

# Initialize Flask-ML server
server = MLServer(__name__)

# Define input schema for image processing
def image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[InputSchema(key="image_input", label="Upload Images", input_type=InputType.BATCHFILE)],
        parameters=[ParameterSchema(
            key="output_path",
            label="Output JSON Path",
            default="output.json",
            value_type=TextParameterDescriptor()
        )]
    )

# Define the TypedDict for image inputs
class ImageInputs(TypedDict):
    image_input: List[BatchFileInput]

class ImageParameters(TypedDict):
    output_path: str

# Helper function to append data to JSON
def append_to_json(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump([data], file, indent=4)

# Main processing function
@server.route("/process_images", task_schema_func=image_task_schema)
def process_images(inputs: ImageInputs, parameters: ImageParameters) -> ResponseBody:
    results = []
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)

    for img_file in inputs["image_input"].files:
        # Indoor/Outdoor detection
        io_result = run_iodetector(img_file.path)
        
        # Geolocation detection
        geo_result = detect_location_from_image(img_file.path, io_result)
        
        # Text detection and language/location extraction
        textspot_results = []
        run_craft(img_file.path, result_folder=temp_folder)
        languages = []
        locations = []

        for processed_img in os.listdir(temp_folder):
            if processed_img.endswith((".jpg", ".png")):
                language, location = get_location_from_text(os.path.join(temp_folder, processed_img))
                languages.append(language)
                locations.extend(location)
        
        # Clean up temporary folder
        for file in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, file))
        
        textspot_results.append({
            "Languages Detected": list(set(languages)),
            "Locations Detected from Text": list(set(locations))
        })

        # Merge results
        geo_result['Languages Detected'] = list(set(languages))
        geo_result['Locations Detected from Text'] = list(set(locations))
        results.append(geo_result)
    
    # Save to output JSON file
    append_to_json(parameters["output_path"], results)
    
    # Remove the temp directory after processing
    shutil.rmtree(temp_folder)

    return ResponseBody(root=BatchTextResponse(texts=[json.dumps(results)]))

# Run the server
if __name__ == "__main__":
    server.run()