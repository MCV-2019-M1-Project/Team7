import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from os.path import isfile, join
from os import listdir
from sklearn import preprocessing
from scipy.spatial import distance
import ml_metrics as metrics
import pickle
from sklearn.metrics import recall_score,precision_score,f1_score

def waterShed(query_img, hue, part):

    ret, thresh = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = 255 - thresh

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # fillin holes
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # Fals. El sure

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.13 * dist_transform.max(), 255, 0)  # 0.15 26.57%

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    markers[markers != 0] = 200

    border_top = int(gray.shape[0] * .03)

    if part != 2:
        border_r = int(gray.shape[1] * .01)
        border_l = int(gray.shape[1] * .02)
    if part != 1:
        border_r = int(gray.shape[1] * .02)
        border_l = int(gray.shape[1] * .01)


    # 35
    markers[0:border_top, :] = 100
    sure_fg[0:border_top, :] = 100

    markers[len(markers) - border_top:len(markers) - 1, :] = 100
    sure_fg[len(markers) - border_top:len(markers) - 1, :] = 100

    markers[:, 0:border_l] = 100
    sure_fg[:, 0:border_l] = 100
    markers[:, len(markers[0, :]) - border_r:len(markers[0, :]) - 1] = 100
    sure_fg[:, len(markers[0, :]) - border_r:len(markers[0, :]) - 1] = 100

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[sure_fg == 0] = 0

    markersres = cv2.watershed(query_img, markers)

    markersres[markersres == 201] = 255
    markersres[markersres == 101] = 0
    markersres[markersres == -1] = 255

    # Make the mask binary
    return markersres / np.max(markersres)

# initialize pixel mask
mask_gt_pixel = np.empty([1])
mask_pred_pixel = np.empty([1])

query2folder = './qsd2_w2/'
histogram_query2_matrix = np.empty([0, 256 * 3])
for filename in sorted(listdir(query2folder)):
    count = 0
    if (filename != '.DS_Store' and (filename.split('.')[1] == 'jpg')):
        # print(query2folder + filename)

        mask_gt = cv2.cvtColor(cv2.imread(query2folder + filename[0:-3] + 'png'), cv2.COLOR_BGR2GRAY) / 255

        query_img = cv2.imread(query2folder + filename)
        gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        hue = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV)[:,:,1]

        if hue.shape[1] >= hue.shape[0]*1.6:
            markersres1 = waterShed(query_img[:, 0:int(hue.shape[1]/2)],hue[:, 0:int(hue.shape[1]/2)], 1)
            markersres2 = waterShed(query_img[:, int(hue.shape[1] / 2)-1:-1], hue[:, int(hue.shape[1] / 2)-1:-1], 2)
            markersres = np.concatenate((markersres1, markersres2), axis=1)

            markersres = cv2.morphologyEx(markersres, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

            markersres = cv2.morphologyEx(markersres, cv2.MORPH_CLOSE, np.ones((1,40), np.uint8), iterations=1)
            markersres = cv2.morphologyEx(markersres, cv2.MORPH_CLOSE, np.ones((40, 1), np.uint8), iterations=1)

        else :
            markersres = waterShed(query_img, hue, 0)
            markersres = cv2.morphologyEx(markersres, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        cv2.imwrite(filename, markersres*255)

        color = ('b', 'g', 'r')

        mask_pred = markersres.astype('uint8')

        mask_gt_pixel = np.concatenate((mask_gt_pixel, mask_gt.flatten()), axis=0)
        mask_pred_pixel = np.concatenate((mask_pred_pixel, mask_pred.flatten()), axis=0)

mask_gt_pixel = mask_gt_pixel[1:]
mask_pred_pixel = mask_pred_pixel[1:]

#PRECISION
precision = precision_score(mask_gt_pixel, mask_pred_pixel)
#RECALL
recall = recall_score(mask_gt_pixel, mask_pred_pixel)
#F1
f1 = f1_score(mask_gt_pixel, mask_pred_pixel)

print('PRECISION: ', precision, 'RECALL: ', recall, ' F1-SCORE: ', f1)