# Team7

## How to run

First of all, these packages should be installed in order to run the code correctly.
- Open CV
- Numpy
- Scipy
- Sklearn
- Mlmetrics
- OS

Folders must be at the same path as the Jupyter Notebook.

The code in the Notebook is divided in sections. Each section can be run independently. Each section consists on a complete task or a part of it, so running a section returns us the results for a concrete task or subtask.  

## Tasks

 - [x] Task1 - Implement 3D / 2D and block and multiresolution histograms
 - [x] Task2 - Test query system using QSD2-W1 (from week1) with Task1 descriptors and evaluate results
 - [x] Task3 - Detect and remove text from images in QSD1-W2
 - [x] Task4 - Evaluate text detection using bounding box (mean IoU)
 - [x] Task5 - Test query system using query set QSD1-W2 development, evaluate retrieval results (use your best performing descriptor)
 - [x] Task6 - For QSD2-W2 : detect all the paintings (max 2 per image), remove background and text, apply retrieval system, return correspondences for each painting. Only retrieval is evaluated.

### Task 1

We divide the image in blocks and compute the histogram for each block. We use different sizes for each block, what we define as the number of levels (a higher value means smaller blocks). 

In that task we have two different implementations:

- Using only one level (a concrete number of blocks, all with the same size) and concatenating the histograms for each block.
- Using different levels and concatenating the histograms for each block and for each level, obtaining a more complex descriptor. 

As the dataset images contain background, we need to remove the background for each image before performing that. 

### Task 2

We test the retrieval system using the Bray Curtis distance as a distance measure. We do not perform any other improvements with respect on what we did the last week. 

### Task 3

We detect the text on the images and obtain the bounding box coordinates for the text. 

This is done by converting the image to the HSV color space and binarize the image using the information from the Saturation channel. After that we apply some morphological filters to finally obtain the mask. 

### Task 4

We analyze how good is the detection of the bounding boxes. To do that we compare the coordinates of our bounding boxes with respect to the ground truth using the mean IoU metric.

### Task 5

With the best descriptors and parameters obtained in the other tasks to compute the histograms of the image, we test the system but ignoring the regions that contains the text (the region inside the bounding box coordinates). 

### Task 6

We detect all the paintings present in the image (no more than 2), remove the background and text obtaining a mask and return the correspondences for each painting. 

To detect all the paintings we first take the information from the Saturation channel and apply the watershed algorithm to it. Then we regularize the mask detecting the corners, and generating and filling a polygon with the corners as vertexs. Then we perform the same as in task 3 to obtain the regions that contain text in the images to obtain the final mask. 

Finally, using the mask we have obtained and the best descriptors and parameters obtained in previous tasks we test the system to obtain the final results. 



