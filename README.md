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

 - [x] Task1 - Detect keypoints and compute descriptors in Museum and query images
 - [x] Task2 - Find tentative matches based on similarity of local appearance and verify matches
 - [x] Task3 - Evaluate the system on QSD1-W4, map@k
 - [x] Task4 - Evaluate best system from previous week on QSD1-W4

### Task 1

We compute SIFT descriptors for every image in the dataset, that have been previously resized. We compute SIFT descriptors for every image in the QSD1_W4, which have also been resized. 

### Task 2

Find matches between the images based on the descriptors that we have computed in Task 1 and discard outliers using the RANSAC algorithm. Then we count the number of matches we have found for every image to provide the retrieved image. 

### Task 3

We test the retrieval system using the QSD1_W4 dataset using different types of descriptor: SIFT, SURF, ORB. The results are provided using the map@k metric.

### Task 4

We test the best retrieval system from Week 3 in the QSD1_W4 dataset. The best method from previous week is using the information of the texture with descriptors of HoG and also using the information of the text. The results are presented using the map@k metric.






