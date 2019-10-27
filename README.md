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

 - [x] Task1 - Filter noise with linear or non-linear filters
 - [x] Task2 - Test query system using QSD1-W2 using only text and a similarity metric to compare text 
 - [x] Task3 - Implement texture descriptors and Test query system using QSD1-W2 using only texture descriptors
 - [x] Task4 - Combine descriptors (Text+Texture, Color+Texture, Text+Color, Text+Texture+Color)
 - [x] Task5 - Repeat the previous analysis for QSD2-W2: remove noise, remove background, find 1 or 2 paintings per image, return correspondences for each painting.

### Task 1

We filter each image on QSD1_W3. Then we compute the sigma for each filtered image. If sigma is bigger than un certain threshold, the image would have noise.
So, only for those images with noise, we apply a bilateral filter in order to reduce the noise and in order to remove it completely, we apply denoise method.

### Task 2

In task 2, we should detect the bounding box on each image, to do it the steps are the following:
 - Resize all the images.
 - Remove words applying opening and closing using a rectangle as a SE.
 - Apply K-means.
 - Crop the images on the top and the bottom obtaining 2 resultant images.
 - Look for uniform regions which will be in the middle and similar to the shape of a rectangle.

After those steps, we choose the shape which is more similar to a rectangle

Ones the bounding box is detected, we apply OCR to detect the text only on the part of the image which is in the bounding box. On this way, we obtain string for each image.

In order to evaluate this method, we use the Levenshtein distance to compare strings.

### Task 3

On this section we implement 3 different types of descriptors, DCT, HoG and LBP. 

To obtain the feature vector using LBP, we divide the image in blocks and then apply LBP for each block. While, in order to obtain the feature vector using DCT and LBP, we use the hole image.

To evaluate these methods, we use the BrayCurtis distance.

The conclusion we obtain on this section is that HoG works much better than the other 2 descriptors proposed.  

### Task 4

On this task, we combined information of color, texture, text. The information of the texture is computed using HoG and the information of the color is obtained using histograms. 

### Task 5

Finally, on this section, we use the code done in Task 1, and the best descriptor and in addition a method to remove background in order to obtain nice results on images from 'QSD2_W3'. 


