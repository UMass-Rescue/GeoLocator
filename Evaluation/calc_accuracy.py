import numpy as np
import pandas as pd
import os
import math
#def top5():

def get_accuracy(predictions):
    if len(predictions)!= 0:
        accuracy = sum(predictions) / len(predictions)
    else:
        accuracy = 1.0
    return accuracy


def top1(results, gt_file):
    io_accuracy = []
    state_accuracy = []
    geoclip_country_Accuracy = []
    geoclip_country_Accuracy_top5 = []
    textspotter_accuracy = []
    scenetype_Accuracy = []
    scenetype_Accuracy_top5 = []
    Language_Accuracy = []
    Location_from_Language_Accuracy = []



    gt = pd.read_csv(gt_file)
    for result in results:
        print(result['Image'].split("\\")[-1])
        #print("Result",result['Environment Type'])
        gt_filter= gt[gt['Image File']==result['Image'].split("\\")[-1]]
        io_accuracy.append(gt_filter['Indoor/ Outdoor'].item()==result['Environment Type'])
        scenes = result['Scene Category']
        scene_list = []
        flag = True
        #print(scenes)
        for scene in scenes:
            scene_list.append(scene['Description'])
            if flag == True and  scene_list[-1] == gt_filter['Type of Scene'].item():
                scenetype_Accuracy_top5.append(True)
                flag = False
        if flag == True:
            scenetype_Accuracy_top5.append(False)
        scenetype_Accuracy.append(gt_filter['Type of Scene'].item()==scene_list[0])


        countries = result['GeoClip Predictions']
        country_list = []
        flag = True
        #print(scenes)
        for country in countries:
            country_list.append(country['Country'])
            if flag == True and  country_list[-1] == gt_filter['Country'].item():
                geoclip_country_Accuracy_top5.append(True)
                flag = False
        if flag == True:
            geoclip_country_Accuracy_top5.append(False)
        geoclip_country_Accuracy.append(gt_filter['Country'].item()==country_list[0])



        
        #print(len(result["Languages Detected Method 2"]))
        #print(gt_filter['Language'].item(),type(gt_filter['Language'].item()),math.isnan(gt_filter['Language'].item()))
        if isinstance(gt_filter['Language'].item(),str):
            lg = str(gt_filter['Language'].item()).split(", ")
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
                    #print(language)
                    if language not in lg:
                        #Language_Accuracy.append(True)
                    #else:
                        Language_Accuracy.append(False)

            #print("Ldvbjfdvb",Language_Accuracy)
            if gt_filter['Country'].item() in result["Countries Possible from Languages Spotted"]:
                Location_from_Language_Accuracy.append(True)
            else:
                Location_from_Language_Accuracy.append(False)
            #print("LL::",Location_from_Language_Accuracy)





            # else:
            #     for language in result["Languages Detected Method 2"]:
            #         if language in gt_filter['Language']:
                        




            


        

            


        


        #print(result)
        print(io_accuracy,scenetype_Accuracy,scenetype_Accuracy_top5,textspotter_accuracy,Language_Accuracy,Location_from_Language_Accuracy,geoclip_country_Accuracy,geoclip_country_Accuracy_top5)
    print("IO Accuracy:",get_accuracy(io_accuracy))
    print("Scene Accuracy:",get_accuracy(scenetype_Accuracy))
    print("Scene Top 5 Accuracy:",get_accuracy(scenetype_Accuracy_top5))
    print("Text Spotter Accuracy", get_accuracy(textspotter_accuracy))
    print("Language Accuracy", get_accuracy(Language_Accuracy))
    print("Location from Language Accuracy", get_accuracy(Location_from_Language_Accuracy))
    print("GeoClip Accuracy",get_accuracy(geoclip_country_Accuracy))
    print("GeoClip TOP 5 Accuracy",get_accuracy(geoclip_country_Accuracy_top5))


    




    