# -*- coding: utf-8 -*-
import sys
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
import cv2
import numpy as np
from TextSpotter.Craft import craft_utils
from TextSpotter.Craft import imgproc
from TextSpotter.Craft import file_utils
from TextSpotter.Craft.craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def save_crops(polys, image, result_folder, filename):
    for i, poly in enumerate(polys):
        poly = np.array(poly).reshape(-1, 2)
        rect = cv2.boundingRect(poly)
        x, y, w, h = rect
        croped = image[y:y+h, x:x+w, :].copy()
        cv2.imwrite(f"{result_folder}/res_{filename}_croped_{i}.png", croped[:, :, ::-1])

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, mag_ratio, canvas_size, refine_net=None):
    # Preprocessing
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    # Make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Refine link if needed
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys

def run_craft(image_path, result_folder='./result', trained_model='weights/craft_mlt_25k.pth',
              text_threshold=0.3, low_text=0.3, link_threshold=0.4, cuda=True, canvas_size=1280,
              mag_ratio=1.5, poly=False, refine=False, refiner_model='weights/craft_refiner_CTW1500.pth'):

    # Load model
    net = CRAFT()
    print('Loading weights from checkpoint (' + trained_model + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    # Load refiner model if required
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
        refine_net.eval()
        poly = True

    # Load image
    image = imgproc.loadImage(image_path)
    filename, _ = os.path.splitext(os.path.basename(image_path))

    # Run text detection
    bboxes, polys = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, mag_ratio, canvas_size, refine_net)

    # Save results
    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
    save_crops(polys, image, result_folder, filename)

