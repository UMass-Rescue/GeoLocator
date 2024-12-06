# Image Upload & Display
import os
from collections import defaultdict
from io import BytesIO
from tkinter import filedialog

# Heatmap
import folium
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from PIL import Image


def detect_location_from_image(model, img_path, jsonfile):

    # Make predictions
    top_pred_gps, top_pred_prob = model.predict(img_path, top_k=50)

    geoLoc = Nominatim(user_agent="GetLoc")

    finallist = []

    save_predictions = defaultdict(int)
    for i in range(5):
        lat, lon = top_pred_gps[i]
        locname = str(geoLoc.reverse((lat, lon)))
        loc_parts = locname.split(",")

        # Check if there are enough parts in loc_parts
        if len(loc_parts) >= 3:
            key = loc_parts[-1].strip() + ", " + loc_parts[-3].strip()
        elif len(loc_parts) >= 1:
            key = loc_parts[-1].strip()  # Use the last part if only one part is available
        else:
            key = "Unknown Location"

        save_predictions[key] += top_pred_prob[i]

    sorted_predictions = sorted(
        save_predictions.items(), key=lambda x: x[1], reverse=True
    )

    for prediction in sorted_predictions:
        x = {}
        loc_parts = prediction[0].split(",")

        # Handle missing parts gracefully
        if len(loc_parts) >= 2:
            x["Country"] = loc_parts[0].strip()
            x["State"] = loc_parts[1].strip()
        elif len(loc_parts) == 1:
            x["Country"] = loc_parts[0].strip()
            x["State"] = "Unknown State"
        else:
            x["Country"] = "Unknown Country"
            x["State"] = "Unknown State"

        x["Confidence"] = str(np.round(prediction[1], 3))  # Removed `.numpy()` since `prediction[1]` is likely a number

        finallist.append(x)

    print(finallist)
    jsonfile["GeoClip Predictions"] = finallist
    return jsonfile
