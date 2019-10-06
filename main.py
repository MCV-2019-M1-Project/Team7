#IMPORTS
import sys
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

print("")
print("********************************************************************************")
print("TASK1 - Create Museum and query image descriptors (BBDD & QS1)")
print("********************************************************************************")

print("")
print("Choose a color space: ")
print("(1) RGB (BGR)")
print("(2) Lab")
print("(3) HSV")
print("(4) XYZ")
print("(5) YCbCr")
colorSpace_number = input ("Enter a number: ")

def colorSpace(img, num):
    if(num == "1"):
        return img
    elif(num == "2"):
        return cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    elif(num == "3"):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    elif(num == "4"):
        return cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
    elif(num == "5"):
        return cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

print("BBDD --> Normalized histogram descriptors --> histogram_bbdd_matrix")
imagesFolder = './bbdd/' #"./bbdd/"
histogram_bbdd_matrix = np.empty([0, 256*3]) #Creem una matriu buida
# print(histogram_bbdd_matrix.shape)
for filename in sorted(listdir(imagesFolder)):
    if(filename != '.DS_Store'):
        # print(imagesFolder + filename)
        img = cv2.imread(imagesFolder + filename)
        img = colorSpace(img,colorSpace_number)
        
        colorx = ('x','y','z')
        hist_img = np.empty([0,0])
        for i,col in enumerate(colorx):
            hist = cv2.calcHist([img],[i],None,[256],[0,256]) #Calculem histogrames
            # hist = preprocessing.normalize(hist, norm='max') #Normalitzem histogrames
            hist_t = hist.transpose()
            # print(hist_t.shape)
            if i == 0:
                hist_img = hist_t
            else:
                hist_img = np.concatenate((hist_img, hist_t), axis = 1)
            
        cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        histogram_bbdd_matrix = np.vstack((histogram_bbdd_matrix, hist_img))   
# print(histogram_bbdd_matrix)

print("QSD1 --> Normalized histogram descriptors --> histogram_query_matrix")
queryFolder = './qsd1_w1/'
histogram_query_matrix = np.empty([0, 256*3])
for filename in sorted(listdir(queryFolder)):
    if(filename != '.DS_Store' and (filename.split('.')[1] == 'jpg' or filename.split('.')[1] == 'png')):
        # print(queryFolder + filename)
        query_img = cv2.imread(queryFolder + filename)
        query_img = colorSpace(query_img,colorSpace_number)

        color = ('b','g','r')
        hist_query = np.empty([0, 256*3])
        for i,col in enumerate(color):
            histr = cv2.calcHist([query_img],[i],None,[256],[0,256])
            # histr = preprocessing.normalize(histr, norm='max')
            histr = histr.transpose()
            if i == 0:
                hist_query = histr
            else:
                hist_query = np.concatenate((hist_query,histr), axis = 1)
        
        cv2.normalize(hist_query, hist_query, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)       
        histogram_query_matrix = np.vstack((histogram_query_matrix, hist_query))
# print(histogram_query_matrix)

print("")
print("********************************************************************************")
print("TASK2 - Implement / compute similarity measures to compare images")
print("********************************************************************************")

print("")
print("Choose a similarity measure: ")
print("(1) Euclidean distance")
print("(2) L1 distance")
print("(3) X2 distance")
print("(4) Histogram intersection")
print("(5) Hellinger kernel")
print("(6) Bray Curtis")
similarityMeasure_number = input ("Enter a number: ")

# print("Defining: Euclidean distance")
def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)

# print("Defining: L1 distance")
def l1_distance(v1, v2):
    return np.linalg.norm(v1 - v2, ord=1)

# print("Defining: X2 distance")    
def x2_distance(v1, v2):
    return

# print("Defining: Histogram intersection")
def histogram_intersection(v1, v2):
    return

# print("Defining: Hellinger kernel")
def hellinger_kernel(v1, v2):
    return

# print("Defining: Bray Curtis")
def bray_curtis(v1, v2):
    return distance.braycurtis(v1, v2)

def similarityMeasure(num, v1, v2):
    if(num == "1"):
        return euclidean_distance(v1, v2)
    elif(num == "2"):
        return l1_distance(v1, v2)
    elif(num == "3"):
        return x2_distance(v1, v2)
    elif(num == "4"):
        return histogram_intersection(v1, v2)
    elif(num == "5"):
        print("hellinger_kernel")
    elif(num == "6"):
        return bray_curtis(v1, v2)

print("")
print("********************************************************************************")
print("TASK3 -  For each image in QSD1, compute similarities to Museum Images and retrieve top K results")
print("********************************************************************************")


K = 10
dst = np.zeros((30, 279))
matrix_retrieval = np.zeros((30,10))

for query_image_index in range(0,len(histogram_query_matrix)): 
    for bbdd_image_index in range(0, len(histogram_bbdd_matrix)):

        dst[query_image_index,bbdd_image_index] = similarityMeasure(similarityMeasure_number, histogram_query_matrix[query_image_index,:], histogram_bbdd_matrix[bbdd_image_index,:])

    matrix_retrieval[query_image_index,:] = np.argsort(dst[query_image_index,:])[:K]

# print("matrix_retrieval")
# print(matrix_retrieval)

print("")
print("********************************************************************************")
print("TASK4 -  Evaluation")
print("********************************************************************************")

#Convertir idx a list of lists
matrix_retrieval_lst = matrix_retrieval.tolist()

with open('./gt_corresps.pkl', 'rb') as fd:
        ll = pickle.load(fd)
        # print(ll)
gt = np.empty((0,0))

ll_prp = np.zeros((len(ll),1))

for i in range(0,len(ll)):
    ll_prp[i] = ll[i][0]

ll_prp_lst = ll_prp.tolist()

mapak = metrics.average_precision.mapk(ll_prp_lst, matrix_retrieval_lst, k=10)        
print("SCORE: ",mapak*100, "%")

#GUARDAR PICKLE RESULT
pickle_out = open("result_qsd1.pkl", "wb")
pickle.dump(matrix_retrieval_lst, pickle_out)
pickle_out.close()

print("")
print("********************************************************************************")
print("TASK5 -  Background removal using color (QS2). Compute descriptor on foreground (painting)")
print("********************************************************************************")

print("")
print("Choose a similarity measure: ")
print("(1) Euclidean distance")
print("(2) L1 distance")
print("(3) X2 distance")
print("(4) Histogram intersection")
print("(5) Hellinger kernel")
print("(6) Bray Curtis")
similarityMeasure_number = input ("Enter a number: ")

#initialize pixel mask
mask_gt_pixel = np.empty([1])
mask_pred_pixel = np.empty([1])

query2folder = './qsd2_w1/'
histogram_query2_matrix = np.empty([0, 256*3])
for filename in sorted(listdir(query2folder)):
    count = 0
    if(filename != '.DS_Store' and (filename.split('.')[1] == 'jpg')):
        # print(query2folder + filename)

        mask_gt = cv2.cvtColor(cv2.imread(query2folder + filename[0:-3] + 'png'), cv2.COLOR_BGR2GRAY)/255

        query_img = cv2.imread(query2folder + filename)
        gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        #hue = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV)[:,:,1]

        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3) #Fals. El sure 

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),255,0) #0.15 26.57% 

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        markers[markers != 0] = 200
        
        #35
        markers[0:40,:] = 100
        sure_fg[0:40,:] = 100
        markers[len(markers)-41:len(markers)-1,:] = 100
        sure_fg[len(markers)-41:len(markers)-1,:] = 100
        markers[:,0:40] = 100
        sure_fg[:,0:40] = 100
        markers[:,len(markers[0,:])-41:len(markers[0,:])-1] = 100
        sure_fg[:,len(markers[0,:])-41:len(markers[0,:])-1] = 100

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[sure_fg == 0] = 0

        markersres = cv2.watershed(query_img,markers)

        markersres[markersres == 201] = 255
        markersres[markersres == 101] = 0
        markersres[markersres == -1] = 255
        
        #Make the mask binary
        markersres = markersres/np.max(markersres)
              
        color = ('b','g','r')
        
        mask_pred = markersres.astype('uint8')

        cv2.imwrite(filename[0:-3]+'png',mask_pred*255)

        mask_gt_pixel = np.concatenate((mask_gt_pixel,mask_gt.flatten()), axis=0)
        mask_pred_pixel = np.concatenate((mask_pred_pixel, mask_pred.flatten()), axis=0)

        #Compute the histogram
        # query_img = cv2.cvtColor(query_img,cv2.COLOR_BGR2Lab)
        query_img = colorSpace(query_img,colorSpace_number)

        hist_query = np.empty([0, 256*3])
        for i,col in enumerate(color):
            histr = cv2.calcHist([query_img],[i],mask_pred,[256],[0,256])
            #Discount the number of black pixels from the corresponding bin on the histogram. 
            histr = histr.transpose()
            if i == 0:
                hist_query = histr
            else:
                hist_query = np.concatenate((hist_query,histr), axis = 1)
        
        cv2.normalize(hist_query, hist_query, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)       
        #Save to the histogram_query matrix      
        histogram_query2_matrix = np.vstack((histogram_query2_matrix, hist_query))


#For each image in QSD2, compute similarities to Museum Images and retrieve top K results
dst = np.zeros((30, 279))
matrix_retrieval = np.zeros((30,10))

for query_image_index in range(0,len(histogram_query2_matrix)): 
    for bbdd_image_index in range(0, len(histogram_bbdd_matrix)):
        
        dst[query_image_index,bbdd_image_index] = similarityMeasure(similarityMeasure_number, histogram_query2_matrix[query_image_index,:], histogram_bbdd_matrix[bbdd_image_index,:])
    
    matrix_retrieval[query_image_index,:] = np.argsort(dst[query_image_index,:])[:10]

# print(matrix_retrieval)



print("")
print("********************************************************************************")
print("TASK6 -  EVALUATION")
print("********************************************************************************")

mask_gt_pixel = mask_gt_pixel[1:]
mask_pred_pixel = mask_pred_pixel[1:]

#PRECISION
precision = precision_score(mask_gt_pixel, mask_pred_pixel)
#RECALL
recall = recall_score(mask_gt_pixel, mask_pred_pixel)
#F1
f1 = f1_score(mask_gt_pixel, mask_pred_pixel)

print('PRECISION: ', precision, 'RECALL: ', recall, ' F1-SCORE: ', f1)

#RETRIEVAL EVALUATION
#Convertir idx(numpy) a list of lists
matrix_retrieval_lst = matrix_retrieval.tolist()
#print(matrix_retrieval_lst)


with open('./qsd2_w1/gt_corresps.pkl', 'rb') as fd:
        ll = pickle.load(fd)
        # print(ll)

ll_prp = np.zeros((len(ll),1))

for i in range(0,len(ll)):
    ll_prp[i] = ll[i][0]

ll_prp_lst = ll_prp.tolist()

mapak = metrics.average_precision.mapk(ll_prp_lst, matrix_retrieval_lst, k=10)        

print("")
print("SCORE: ", mapak*100, "%")


#GUARDAR PICKLE RESULT
pickle_out = open("result_qsd2.pkl", "wb")
pickle.dump(matrix_retrieval_lst, pickle_out)
pickle_out.close()