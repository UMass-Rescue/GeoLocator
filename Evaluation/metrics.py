import numpy as np
import pandas as pd
import os


def get_average(predictions):
    """Calculate the accuracy from a list of predictions."""
    if len(predictions) != 0:
        accuracy = round(sum(predictions) / len(predictions),2)
    else:
        accuracy = 1.0
    return accuracy

def get_average_ind(predictions):
    """Calculate the accuracy from a list of predictions."""
    predictions["Precision"] =get_average(predictions["Precision"])
    predictions["Recall"] = get_average(predictions["Recall"])

    predictions["F1-Score"] = get_average(predictions["F1-Score"])
    predictions["Accuracy"] = get_average(predictions["Accuracy"])


    return predictions

def calculate_metrics(y_true, y_pred):
    # Dynamically create a list of all unique labels from y_true and y_pred

    y_true_set = set(y_true)
    y_pred_set = set(y_pred)

    true_positive = len(y_true_set & y_pred_set)
    false_positive = len(y_pred_set - y_true_set)
    false_negative = len(y_true_set - y_pred_set)

    # Calculate metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate accuracy
    accuracy = true_positive / len(y_true_set) if len(y_true_set) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy
    }

def append_to_Results(list,current):
    list["Precision"].append(current["Precision"])
    list["Recall"].append(current["Recall"])

    list["F1-Score"].append(current["F1-Score"])
    list["Accuracy"].append(current["Accuracy"])

    return list

def calculate(results, gt_file):
    """Evaluate the top-1 accuracy and other metrics for the results."""

    LD = {"Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Accuracy": []}
    
    SC = {"Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Accuracy": []}
    
    GC = {"Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Accuracy": []}
    
    #IO = []
    # state_accuracy = []
    geoclip_country_Accuracy = []
    geoclip_country_Accuracy_top5 = []
    io_accuracy = []
    textspotter_accuracy = []
    # Scene = []
    # scenetype_Accuracy_top5 = []
    # Language_Accuracy = []
    Location_from_Language_Accuracy = []



    # Load the ground truth file
    gt = pd.read_csv(gt_file)
    for result in results:
        # Extract the image filename using os.path.basename for cross-platform compatibility
        image_filename = os.path.basename(result["Image"])
        #print(image_filename)

        # Match the image filename with the ground truth data
        gt_filter = gt[gt["Image File"] == image_filename]
        if len(gt_filter) == 1:
            gt_row = gt_filter.iloc[0]
            #IO_current = calculate_metrics(gt_row["Indoor/ Outdoor"].split(),result["Environment Type"].split())
            #IO = append_to_Results(IO,IO_current)
            io_accuracy.append(gt_row["Indoor/ Outdoor"] == result["Environment Type"])

            # Scene category accuracy
            scenes = result["Scene Category"]
            scene_list = []
    #         flag = True
            for scene in scenes:
                scene_list.append(scene["Description"])
    #             if flag and scene_list[-1] == gt_row["Type of Scene"]:
    #                 scenetype_Accuracy_top5.append(True)
    #                 flag = False
    #         if flag:
    #             scenetype_Accuracy_top5.append(False)
    #         scenetype_Accuracy.append(gt_row["Type of Scene"] == scene_list[0])
            scene_values = [value.strip() for value in gt_row["Type of Scene"].split(",")]

            SC_current = calculate_metrics(scene_values,scene_list)
            SC = append_to_Results(SC,SC_current)


            #print(IO , "IO")
            #print(SC , "SC")
            
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

    #         # TextSpotter and language detection accuracy
            if isinstance(gt_row["Language"], str):
                lg = str(gt_row["Language"]).split(", ")
                lg_predicted = result.get("Languages Detected Method 2", [])
                if len(lg_predicted)>0:
                    textspotter_accuracy.append(True)
                    if (
                    gt_row["Country"]
                    in result["Countries Possible from Languages Spotted"]
                ):
                        Location_from_Language_Accuracy.append(True)
                    else:
                        Location_from_Language_Accuracy.append(False)
                    #countries_predicted = result.get("Languages Detected Method 2", [])

                    #calculate_metrics(gt_row["Country"].split(),countries_predicted)

                
                else:
                    textspotter_accuracy.append(False)
                LD_current = calculate_metrics(lg,lg_predicted)
                LD = append_to_Results(LD,LD_current)

    print()

    # print("IO Accuracy:", io_accuracy)
    print("Indoor Outdoor Accuracy:", get_average(io_accuracy))

    #print("Scene Accuracy:", SC)
    print("Scene Accuracy:", get_average_ind(SC))

    #print("Scene Top 5 Accuracy:", get_accuracy(scenetype_Accuracy_top5))
    #print("Text Spotter Accuracy", textspotter_accuracy)
    print("Text Spotter Accuracy", get_average(textspotter_accuracy))

    #print("Language Accuracy", LD)
    print("Language Accuracy", get_average_ind(LD))

    #print("Location from Language Accuracy", Location_from_Language_Accuracy)
    print("Location from Language Accuracy", get_average(Location_from_Language_Accuracy))
    #print("GeoClip Accuracy", geoclip_country_Accuracy)
    print("GeoClip Accuracy", get_average(geoclip_country_Accuracy))
    #print("GeoClip TOP 5 Accuracy", geoclip_country_Accuracy_top5)
    print("GeoClip TOP 5 Accuracy", get_average(geoclip_country_Accuracy_top5))