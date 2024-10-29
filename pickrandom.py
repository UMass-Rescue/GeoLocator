import zipfile
import os
from PIL import Image
from io import BytesIO
import random

def extract_n_images(zip_path, output_folder, n=5):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Filter out image files (png, jpg, jpeg, etc.)
        image_files = [file for file in zip_ref.namelist() if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)
        
        # Limit to n images
        selected_files = image_files[:n]
        print(selected_files)
        
        for file in selected_files:
            # Read the image file from zip
            with zip_ref.open(file) as image_file:
                # Convert it to an Image object
                image = Image.open(BytesIO(image_file.read()))
                if file.split("/")[1]=='Exterior':
                    # Save the image to the output folder
                    output_path = os.path.join(output_folder,os.path.splitext(os.path.basename(file))[0]+'_EXT.png')
                elif file.split("/")[1]=='Interior':
                    output_path = os.path.join(output_folder,os.path.splitext(os.path.basename(file))[0]+'_INT.png')
                else:
                    print("Error with the file names, please check")
                image.save(output_path)
                
    print(f"{len(selected_files)} images extracted to '{output_folder}'")

# Usage
extract_n_images('IndoorOutdoorClassifier/dataset.zip', 'testImages', n=10)
