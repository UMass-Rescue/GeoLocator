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
def detect_location_from_image(img_path):
    model = GeoCLIP()

    # Make predictions
    top_pred_gps, top_pred_prob = model.predict(img_path, top_k=50)

    geoLoc = Nominatim(user_agent="GetLoc")


    # Display the top 5 GPS predictions
    print("Top 10 GPS Predictions 📍")
    print("========================")

    save_predictions = defaultdict(int)
    for i in range(5):
        lat, lon = top_pred_gps[i]
        #print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f}) - Probability: {top_pred_prob[i]:.6f}")
        locname = str(geoLoc.reverse((lat, lon)))
        #print(locname)
        save_predictions[locname.split(",")[-1] + ", " + locname.split(",")[-3]] += top_pred_prob[i]
        #print(locname.split(",")[-1] + ", " + locname.split(",")[-2])
    print(len(save_predictions),save_predictions)
        #sorted_predictions


