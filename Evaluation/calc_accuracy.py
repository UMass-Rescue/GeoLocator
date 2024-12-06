import math
import os

import numpy as np
import pandas as pd


def get_accuracy(predictions):
    """Calculate the accuracy from a list of predictions."""
    if len(predictions) != 0:
        accuracy = sum(predictions) / len(predictions)
    else:
        accuracy = 1.0
    return accuracy


def top1(results, gt_file):
    """Evaluate the top-1 accuracy and other metrics for the results."""
    io_accuracy = []
    state_accuracy = []
    geoclip_country_Accuracy = []
    geoclip_country_Accuracy_top5 = []
    textspotter_accuracy = []
    scenetype_Accuracy = []
    scenetype_Accuracy_top5 = []
    Language_Accuracy = []
    Location_from_Language_Accuracy = []

    # Load the ground truth file
    gt = pd.read_csv(gt_file)
    for result in results:
        # Extract the image filename using os.path.basename for cross-platform compatibility
        image_filename = os.path.basename(result["Image"])
        print(image_filename)

        # Match the image filename with the ground truth data
        gt_filter = gt[gt["Image File"] == image_filename]
        if len(gt_filter) == 1:
            gt_row = gt_filter.iloc[0]
            io_accuracy.append(
                gt_row["Indoor/ Outdoor"] == result["Environment Type"]
            )

            # Scene category accuracy
            scenes = result["Scene Category"]
            scene_list = []
            flag = True
            for scene in scenes:
                scene_list.append(scene["Description"])
                if flag and scene_list[-1] == gt_row["Type of Scene"]:
                    scenetype_Accuracy_top5.append(True)
                    flag = False
            if flag:
                scenetype_Accuracy_top5.append(False)
            scenetype_Accuracy.append(gt_row["Type of Scene"] == scene_list[0])

            # GeoClip country accuracy
            countries = result["GeoClip Predictions"]
            country_list = []
            flag = True
            for country in countries:
                country_list.append(country["Country"])
                if flag and country_list[-1] == gt_row["Country"]:
                    geoclip_country_Accuracy_top5.append(True)
                    flag = False
            if flag:
                geoclip_country_Accuracy_top5.append(False)
            geoclip_country_Accuracy.append(gt_row["Country"] == country_list[0])

            # TextSpotter and language detection accuracy
            if isinstance(gt_row["Language"], str):
                lg = str(gt_row["Language"]).split(", ")
                if len(lg) <= len(result.get("Languages Detected Method 2", [])):
                    textspotter_accuracy.append(True)
                else:
                    textspotter_accuracy.append(False)

                for language in lg:
                    print(language)
                    if language in result["Languages Detected Method 2"]:
                        Language_Accuracy.append(True)
                    else:
                        Language_Accuracy.append(False)
                for language in result["Languages Detected Method 2"]:
                    if language not in lg:
                        Language_Accuracy.append(False)

                if (
                    gt_row["Country"]
                    in result["Countries Possible from Languages Spotted"]
                ):
                    Location_from_Language_Accuracy.append(True)
                else:
                    Location_from_Language_Accuracy.append(False)
        else:
            print(f"No match or multiple matches found for: {image_filename}")
            io_accuracy.append(False)

    print(
        io_accuracy,
        scenetype_Accuracy,
        scenetype_Accuracy_top5,
        textspotter_accuracy,
        Language_Accuracy,
        Location_from_Language_Accuracy,
        geoclip_country_Accuracy,
        geoclip_country_Accuracy_top5,
    )

    print("IO Accuracy:", get_accuracy(io_accuracy))
    print("Scene Accuracy:", get_accuracy(scenetype_Accuracy))
    print("Scene Top 5 Accuracy:", get_accuracy(scenetype_Accuracy_top5))
    print("Text Spotter Accuracy", get_accuracy(textspotter_accuracy))
    print("Language Accuracy", get_accuracy(Language_Accuracy))
    print(
        "Location from Language Accuracy", get_accuracy(Location_from_Language_Accuracy)
    )
    print("GeoClip Accuracy", get_accuracy(geoclip_country_Accuracy))
    print("GeoClip TOP 5 Accuracy", get_accuracy(geoclip_country_Accuracy_top5))
