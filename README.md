# Team7

## How to run

First of all, these packages should be installed in order to run the code correctly.
- Open CV
- Numpy
- Scipy
- Sklearn
- Mlmetrics
- OS

Folders must be at the same path as main.py.

In order to execute the code, go to the terminal and execute the following: `$ python3 main.py`

At the begining the user should choose between different space colors:   
(1) RGB (BGR)  
(2) Lab  
(3) HSV  
(4) XYZ  
(5) YCbCr  

After that, chosen between different similarity metric:  
(1) Euclidean distance  
(2) L1 distance  
(3) X2 distance (Not aviable)  
(4) Histogram intersection (Not aviable)  
(5) Hellinger kernel (Not aviable)  
(6) Bray Curtis  

After selectin the previous options it will appear the result of the Query dataset 1 and the Query dataset 2 join with the Precision, Recall and F1. 

## W2
 - [x] Task1 - Implement 3D / 2D and block and multiresolution histograms
 - [x] Task2 - Test query system using QSD2-W1 (from week1) with Task1 descriptors and evaluate results
 - [ ] Task3 - Detect and remove text from images in QSD1-W2
 - [ ] Task4 - Evaluate text detection using bounding box (mean IoU)
 - [ ] Task5 - Test query system using query set QSD1-W2 development, evaluate retrieval results (use your best performing descriptor)
 - [ ] Task6 - For QSD2-W2 : detect all the paintings (max 2 per image), remove background and text, apply retrieval system, return correspondences for each painting. Only retrieval is evaluated.
