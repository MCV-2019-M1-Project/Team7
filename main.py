#IMPORTS
import sys
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from os.path import isfile, join
from os import listdir
from sklearn import preprocessing

print("")
print("********************************************************************************")
print("TASK1 - Create Museum and query image descriptors (BBDD & QS1)")
print("********************************************************************************")
print("BBDD --> Normalized histogram descriptors --> histogram_bbdd_matrix")
imagesFolder = "./bbdd/"
histogram_bbdd_matrix = np.empty([0, 256*3]) #Creem una matriu buida
#print(histogram_bbdd_matrix.shape)
for filename in sorted(listdir(imagesFolder)):
    if(filename != '.DS_Store' ):
        # print("./qsd1_w1/" + filename)
        img = cv2.imread(imagesFolder + filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
        color = ('b','g','r')
        hist_img = np.empty([0,0])
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256]) #Calculem histogrames
            hist = preprocessing.normalize(hist, norm='l2') #Normalitzem histogrames
            hist_t = hist.transpose()
            # print(hist_t.shape)
            if i == 0:
                hist_img = hist_t
            else:
                hist_img = np.concatenate((hist_img, hist_t), axis = 1)
 
        histogram_bbdd_matrix = np.vstack((histogram_bbdd_matrix, hist_img)) 
print(histogram_bbdd_matrix)

print("QSD1 --> Normalized histogram descriptors --> histogram_query_matrix")
queryFolder = "./qsd1_w1/"
histogram_query_matrix = np.empty([0, 256*3])
for filename in sorted(listdir(queryFolder)):
    if(filename != '.DS_Store' and (filename.split('.')[1] == 'jpg' or filename.split('.')[1] == 'png')):
        # print("./qsd1_w1/" + filename)
        query_img = cv2.imread(queryFolder + filename)
        query_img = cv2.cvtColor(query_img,cv2.COLOR_BGR2Lab)
        color = ('b','g','r')
        hist_query = np.empty([0, 256*3])
        
        for i,col in enumerate(color):
            histr = cv2.calcHist([query_img],[i],None,[256],[0,256])
            histr = preprocessing.normalize(histr, norm='l2')
            histr = histr.transpose()
            if i == 0:
                hist_query = histr
            else:
                hist_query = np.concatenate((hist_query,histr), axis = 1)
        
        histogram_query_matrix = np.vstack((histogram_query_matrix, hist_query))
print(histogram_query_matrix)

print("")
print("********************************************************************************")
print("TASK2 - Implement / compute similarity measures to compare images")
print("********************************************************************************")

print("Defining: Euclidean distance")
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2, ord=1)

print("Defining: L1 distance")
def l1_distance(v1, v2):
    return

print("Defining: X2 distance")    
def x2_distance(v1, v2):
    return

print("Defining: Histogram intersection")
def histogram_intersection(v1, v2):
    return

print("Defining: Hellinger kernel")
def hellinger_kernel(v1, v2):
    return

print("")
print("********************************************************************************")
print("TASK3 -  For each image in QS1, compute similarities to Museum Images and retrieve top K results")
print("********************************************************************************")

from scipy.spatial import distance

dst = np.zeros((30, 279))
matrix_retrieval = np.zeros((30,10))

for query_image_index in range(0,len(histogram_query_matrix)): 
    for bbdd_image_index in range(0, len(histogram_bbdd_matrix)):
        
        euclidean_distance(histogram_query_matrix[query_image_index,:], histogram_bbdd_matrix[bbdd_image_index,:])
        
    matrix_retrieval[query_image_index,:] = np.argsort(dst[query_image_index,:])[:10]

print(matrix_retrieval)

print("")
print("********************************************************************************")
print("TASK4 -  Evaluation")
print("********************************************************************************")

import ml_metrics as metrics
import pickle

with open('./qsd1_w1/gt_corresps.pkl', 'rb') as fd:
        ll = pickle.load(fd)
        print(ll)
gt = np.empty((0,0))

#Convertir idx a list of lists
matrix_retrieval_lst = matrix_retrieval.tolist()

ll_prp = np.zeros((len(ll),1))

for i in range(0,len(ll)):
    ll_prp[i] = ll[i][0][1]

ll_prp_lst = ll_prp.tolist()

mapak = metrics.average_precision.mapk(ll_prp_lst, matrix_retrieval_lst, k=10)        

print(mapak)