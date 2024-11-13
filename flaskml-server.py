import json
import os
import shutil
from typing import List, TypedDict
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (BatchFileInput, BatchTextResponse,
                                             FileInput, FileResponse, FileType,
                                             InputSchema, InputType,
                                             NewFileInputType, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor)
from geoclipModule.geoclip import detect_location_from_image
from IndoorOutdoorClassifier.iodetector import run_iodetector
from TextSpotter.Craft.textspot import run_craft
from utils.textExtraction import get_location_from_text

# Initialize Flask-ML server
server = MLServer(__name__)

server.add_app_metadata(
    name="GeoLocator",
    author="Islam",
    version="0.1.0",
    info=load_file_as_string("./README.md"),
)

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

class ImageInputs(TypedDict):
    image_input: BatchFileInput
    output_path: FileInput

class ImageParameters(TypedDict): ...

def append_to_json(file_path, data):
    if os.path.exists(file_path):
        with open(file_path, "r+") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
            existing_data.append(data) if isinstance(existing_data, list) else [existing_data, data]
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, "w") as file:
            json.dump([data], file, indent=4)

@server.route("/process_images", task_schema_func=image_task_schema, short_title="Result")
def process_images(inputs: ImageInputs, parameters: ImageParameters) -> ResponseBody:
    results = []
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    
    for img_file in inputs["image_input"].files:
        print("Processing image:", img_file.path)
        
        io_result = run_iodetector(img_file.path)
        print("IO Detection Result:", io_result)
        
        geo_result = detect_location_from_image(img_file.path, io_result)
        print("Geo Detection Result:", geo_result)

        textspot_results = []
        try:
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

        languages, locations = [], []
        for processed_img in os.listdir(temp_folder):
            if processed_img.endswith((".jpg", ".png")):
                processed_path = os.path.join(temp_folder, processed_img)
                try:
                    language, location = get_location_from_text(processed_path)
                    languages.append(language)
                    locations.extend(location)
                except Exception as e:
                    print(f"Error processing {processed_path}: {e}")

        textspot_results.append({
            "Languages Detected": list(set(languages)),
            "Locations Detected from Text": list(set(locations)),
        })

        geo_result["Languages Detected"] = list(set(languages))
        geo_result["Locations Detected from Text"] = list(set(locations))
        results.append(geo_result)

    try:
        append_to_json(inputs["output_path"].path, results)
        print("Results written to:", inputs["output_path"].path)
    except Exception as e:
        print(f"Failed to write results: {e}")

    shutil.rmtree(temp_folder)

    return ResponseBody(
        root=FileResponse(
            file_type=FileType.JSON,
            path=inputs["output_path"],
            title="Report",
        )
    )

if __name__ == "__main__":
    server.run()
