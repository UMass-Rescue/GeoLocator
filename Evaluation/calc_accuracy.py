import numpy as np
import pandas as pd
import os

#def top5():

def get_accuracy(predictions):
    accuracy = sum(predictions) / len(predictions)
    return accuracy


def top1(results, gt_file):
    io_accuracy = []
    state_accuracy = []
    geoclip_country_Accuracy = []
    textspotter_accuracy = []
    scenetype_Accuracy = []
    Language_Accuracy = []



    gt = pd.read_csv(gt_file)
    for result in results:
        print(result['Image'].split("\\")[-1])
        print("Result",result['Environment Type'])
        gt_filter= gt[gt['Image File']==result['Image'].split("\\")[-1]]
        print(gt_filter)
        #geoclip_country_Accuracy.append(gt['Country']==result[''])
        print(gt_filter['Indoor/ Outdoor'].item())
        io_accuracy.append(gt_filter['Indoor/ Outdoor'].item()==result['Environment Type'])


        print(result)
    print(get_accuracy(io_accuracy))



    