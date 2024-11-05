from geoclip import GeoCLIP
# Image Upload & Display
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os
from tkinter import filedialog

import numpy as np
# Heatmap
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from collections import defaultdict
def detect_location_from_image(img_path,jsonfile):
    model = GeoCLIP()

    # Make predictions
    top_pred_gps, top_pred_prob = model.predict(img_path, top_k=50)

    geoLoc = Nominatim(user_agent="GetLoc")
    
    finallist = [{"Country, State" : "Confidence"}]
     
    # Display the top 5 GPS predictions
    #print("Top 10 GPS Predictions 📍")
    #print("========================")

    save_predictions = defaultdict(int)
    for i in range(10):
        lat, lon = top_pred_gps[i]
        #print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f}) - Probability: {top_pred_prob[i]:.6f}")
        locname = str(geoLoc.reverse((lat, lon)))
        #print(locname)
        save_predictions[locname.split(",")[-1] + ", " + locname.split(",")[-3]] += top_pred_prob[i]
        #print(locname.split(",")[-1] + ", " + locname.split(",")[-2])
    #print(len(save_predictions),save_predictions)
    sorted_predictions = sorted(save_predictions.items(), key=lambda x:x[1],reverse=True)
    #print(sorted_predictions)
    x = {}
    for prediction in sorted_predictions:
        x[prediction[0]]=str(np.round(prediction[1].numpy(),3))
    
    #print(finallist)
    finallist.append(x)
    #print(finallist)
    jsonfile['GeoClip Predictions'] = finallist
    #print(jsonfile)
    return jsonfile


