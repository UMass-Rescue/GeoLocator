import argparse
import os
from IndoorOutdoorClassifier.iodetector import run_iodetector
from geoclipModule import geoclip 
import json
import warnings
from TextSpotter.MMOCR.tools import dba
warnings.filterwarnings("ignore")

def append_to_json(file_path, data):
    """Appends data to a JSON file, creating it if it doesn't exist."""

    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
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
        with open(file_path, 'w') as file:
            json.dump([data], file, indent=4)

def start(directory,outputJson):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]  # Add more if needed
    image_files = []
    #directory = args.image_folder
    for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                if any(filename.endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(directory, filename))


    print(f"There are {len(image_files)} images in the folder selected")
    for img in image_files:
         op = run_iodetector(img)
         geoclipOp = geoclip.detect_location_from_image(img,op)
         #dba.infer(inputs=img)
         append_to_json(outputJson, geoclipOp)




def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    parser.add_argument("--image_folder", help="folder to be tested for locations",type = str,default='testImages')
    parser.add_argument("--output_csv", help="folder to be tested for locations",type = str,default='op.json')
    args = parser.parse_args()
    start(args.image_folder,args.output_csv)
    #return args

    



if __name__ == "__main__":
    main()

