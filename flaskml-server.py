from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    BatchFileInput, BatchTextResponse, ResponseBody, TaskSchema, InputSchema, InputType
)
from geoclipModule.geoclip import detect_location_from_image
from IndoorOutdoorClassifier.iodetector import run_iodetector
from TextSpotter.Craft.textspot import run_craft
from utils.textExtraction import get_location_from_text
from typing import TypedDict  # Import TypedDict
import os

# Initialize Flask-ML server
server = MLServer(__name__)

# Define the schema for endpoints that accept batch file input (images)
def image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[InputSchema(key="image_input", label="Upload Images", input_type=InputType.BATCHFILE)],
        parameters=[]
    )

# Define the input format for the `/geoclip`, `/iodetector`, `/textspotter`, and `/text_extraction` endpoints
class ImageInputs(TypedDict):
    image_input: BatchFileInput

# Define an empty TypedDict for the `parameters` argument
class EmptyParameters(TypedDict):
    pass

# GeoClip endpoint
@server.route("/geoclip", task_schema_func=image_task_schema)
def geoclip_endpoint(inputs: ImageInputs, parameters: EmptyParameters) -> ResponseBody:
    results = []
    for img_file in inputs["image_input"].files:
        result = detect_location_from_image(img_file.path, {})
        results.append(result)
    return ResponseBody(root=BatchTextResponse(texts=results))

# IO Detector endpoint
@server.route("/iodetector", task_schema_func=image_task_schema)
def iodetector_endpoint(inputs: ImageInputs, parameters: EmptyParameters) -> ResponseBody:
    results = []
    for img_file in inputs["image_input"].files:
        result = run_iodetector(img_file.path)
        results.append(result)
    return ResponseBody(root=BatchTextResponse(texts=results))

# TextSpotter endpoint
@server.route("/textspotter", task_schema_func=image_task_schema)
def textspotter_endpoint(inputs: ImageInputs, parameters: EmptyParameters) -> ResponseBody:
    results = []
    temp_folder = "temp_results"
    os.makedirs(temp_folder, exist_ok=True)
    
    for img_file in inputs["image_input"].files:
        run_craft(img_file.path, result_folder=temp_folder)
        
        locations, languages = [], []
        for processed_img in os.listdir(temp_folder):
            if processed_img.endswith((".jpg", ".png")):
                lang, loc = get_location_from_text(os.path.join(temp_folder, processed_img))
                languages.append(lang)
                locations.extend(loc)
        
        results.append({
            "Languages Detected": list(set(languages)),
            "Locations Detected from Text": list(set(locations))
        })

    # Clean up temporary folder
    for file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file))
    os.rmdir(temp_folder)

    return ResponseBody(root=BatchTextResponse(texts=results))

# Text Extraction endpoint
@server.route("/text_extraction", task_schema_func=image_task_schema)
def text_extraction_endpoint(inputs: ImageInputs, parameters: EmptyParameters) -> ResponseBody:
    results = []
    for img_file in inputs["image_input"].files:
        lang, locations = get_location_from_text(img_file.path)
        results.append({
            "Detected Language": lang,
            "Recognized Locations": locations
        })
    return ResponseBody(root=BatchTextResponse(texts=results))

# Run the server
if __name__ == "__main__":
    server.run()