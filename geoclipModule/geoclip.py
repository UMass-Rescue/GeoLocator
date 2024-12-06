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
        save_predictions[
            locname.split(",")[-1] + ", " + locname.split(",")[-3]
        ] += top_pred_prob[i]
    sorted_predictions = sorted(
        save_predictions.items(), key=lambda x: x[1], reverse=True
    )

    for prediction in sorted_predictions:
        x = {}
        x["Country"] = prediction[0].split(",")[0].strip()
        x["State"] = prediction[0].split(",")[1].strip()
        x["Confidence"] = str(np.round(prediction[1].numpy(), 3))

    finallist.append(x)
    print(finallist)
    jsonfile["GeoClip Predictions"] = finallist
    return jsonfile
