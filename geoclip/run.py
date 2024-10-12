from geoclip import GeoCLIP
# Image Upload & Display
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os
from tkinter import filedialog

# Heatmap
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from collections import defaultdict



import torch

if not torch.cuda.is_available():
    raise RuntimeError("This notebook requires a GPU to run. Please ensure that you have selected GPU as the Hardware accelerator in the Runtime settings.")
else:
    print("CUDA is available. You're good to go!")


model = GeoCLIP().to("cuda")
print("===========================")
print("GeoCLIP has been loaded! 🎉")
print("===========================")

file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=(("Image files", "*.jpg;*.png;*.jpeg;*.bmp;*.gif"), ("All files", "*.*"))
    )
print(file_path)

print("===========================")
print("Image selected!. Model trying to predict, meanwhile can you guess??")
print("===========================")

# Make predictions
top_pred_gps, top_pred_prob = model.predict(file_path, top_k=50)

geoLoc = Nominatim(user_agent="GetLoc")


# Display the top 5 GPS predictions
print("Top 10 GPS Predictions 📍")
print("========================")

save_predictions = defaultdict(int)
for i in range(10):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f}) - Probability: {top_pred_prob[i]:.6f}")
    locname = str(geoLoc.reverse((lat, lon)))
    print(locname)
    save_predictions[locname.split(",")[-1] + ", " + locname.split(",")[-3]] += 1
    print(locname.split(",")[-1] + ", " + locname.split(",")[-2])
print(save_predictions)


