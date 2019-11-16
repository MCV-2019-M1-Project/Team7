import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from cv2 import xfeatures2d_SIFT, xfeatures2d_SURF, KeyPoint, Sobel
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import convolve2d
import math
from skimage.feature import canny, hog
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.linear_model import RANSACRegressor as RANSAC
from PIL import Image
import os

import pickle
import ml_metrics as metrics

from skimage.draw import line_aa


def load_imgs(path, ext ="jpg"):
    paths = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f)) and ext in f])
    return [cv2.imread(path) for path in tqdm(paths)]

def resize_to_min(image, min_shape):
    # Image shape
    shape = image.shape[:2]
    # Get smaller axis
    argmin = np.argmax(shape)

    # Compute resize factor
    resize_factor = min_shape[argmin]/shape[argmin]
    # Apply resizing
    resize_size = (int(shape[1]*resize_factor), int(shape[0]*resize_factor))
    return cv2.resize(image, resize_size)

def detect_corners(idx, masks):
    """
    Finds four points corresponding to rectangle corners

    :param mask: (ndarray) binary image
    :return: (int) points from corners
    """

    corners = []
    for num_mask, mask in enumerate(masks):

        width = mask.shape[1]
        height = mask.shape[0]
        coords = np.argwhere(np.ones([height, width]))
        coords_x = coords[:, 1]
        coords_y = coords[:, 0]

        coords_x_filtered = np.extract(mask, coords_x)
        coords_y_filtered = np.extract(mask, coords_y)
        max_br = np.argmax(coords_x_filtered + coords_y_filtered)
        max_tr = np.argmax(coords_x_filtered - coords_y_filtered)
        max_tl = np.argmax(-coords_x_filtered - coords_y_filtered)
        max_bl = np.argmax(-coords_x_filtered + coords_y_filtered)

        tl_x, tl_y = int(coords_x_filtered[max_tl]), int(coords_y_filtered[max_tl])
        tr_x, tr_y = int(coords_x_filtered[max_tr]), int(coords_y_filtered[max_tr])
        bl_x, bl_y = int(coords_x_filtered[max_bl]), int(coords_y_filtered[max_bl])
        br_x, br_y = int(coords_x_filtered[max_br]), int(coords_y_filtered[max_br])

        point1 = (tl_x, tl_y)
        point2 = (tr_x, tr_y)
        point3 = (br_x, br_y)
        point4 = (bl_x, bl_y)
        allCoordinates = [point1, point2, point3, point4]
        corners.append(allCoordinates)

    return corners

def crop(image, mask):

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1 # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    mask_cropped = mask[x0:x1, y0:y1]

    return cropped, mask_cropped, x0, y0

def crop_paintings(idx, images, masks, first_crop):

    cropped_paintings = []
    cropped_masks = []
    for num_mask, mask in enumerate(masks):
        if first_crop:
            mask = cv2.dilate(mask, np.ones((20, 1), np.uint8), iterations=1)
            mask = cv2.dilate(mask, np.ones((1, 20), np.uint8), iterations=1)
            mask = cv2.resize(mask, (images.shape[1], images.shape[0]))
            image = images
        else:
            image = images[num_mask]
        cropped, mask_cropped, _, _ = crop(image, mask)
        cropped_paintings.append(cropped)
        cropped_masks.append(mask_cropped)
        #name = "./cropped_paintings/cropped_" + str(idx) + "_" + str(num_mask) + ".jpg"
        #cv2.imwrite(name, cropped)

    return cropped_paintings, cropped_masks

def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr = 8.0
    h, w = gray.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(gray, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (w - 2) * (h - 2))
    if sigma > thr:
        return True
    else:
        return False

def remove_noise(image):
    if detect_noise(image):
        image_smooth = cv2.bilateralFilter(image, 9, 120, 120)
        return cv2.fastNlMeansDenoisingColored(image_smooth, None, 12, 10, 7, 21)
    else:
        return image

def find_keypoints(image, descriptor):
    if descriptor is 'SIFT':
        # Initiate SIFT detector
        sift = xfeatures2d_SIFT.create()

        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(image, None)
    elif descriptor is 'ORB':
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

    elif descriptor is 'SURF':
        # Initiate SURF detector
        surf = xfeatures2d_SURF.create()

        # find the keypoints and descriptors with SIFT
        kp, des = surf.detectAndCompute(image, None)

    return kp, des

def find_matches(bbdd,qsd,method,MIN_MATCH_COUNT):
    if method is 'BRUTE_FORCE':
        #print(qsd)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = bf.match(bbdd[1], qsd[1])

    elif method is 'FLANN':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # BFMatcher with default params
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_param = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(bbdd[1], qsd[1], k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([bbdd[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([qsd[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel()

        dist = [m.distance for m, m_mask in zip(good, matchesMask) if m_mask == 1]
        dist = np.sum(np.asarray(dist))/len(dist)

    else:
        matchesMask = 0
        dist = 99999


    numMatches = np.sum(matchesMask)

    return dist

def find_correspondences(bbdd_kp_des, qs_kp_des, method,MIN_MATCH_COUNT = 10):

    dist = [find_matches(bd_kp_des, qs_kp_des, method,MIN_MATCH_COUNT) for bd_kp_des in bbdd_kp_des]

    return dist

def estimate_lines(image, param):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_canny = canny(gray, param[0], param[1], param[2])
    edges = edge_canny

    lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                     line_gap=3)

    return lines

def remove_bg(num_image, image):

    lines = estimate_lines(image, (2, 1, 25))

    image_lines = np.zeros(image.shape[:2], np.uint8)

    for line in lines:
        p0, p1 = line
        r, c, val = line_aa(p0[0], p0[1], p1[0], p1[1])
        image_lines[c, r] = 255

    image_lines_resized = cv2.resize(image_lines, (500, 500))
    image_resized = cv2.resize(image, (500, 500))

    closing = cv2.morphologyEx(image_lines_resized, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)  # 0.15 26.57%

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    markers[markers != 0] = 200

    border_top = int(closing.shape[0] * .03)
    borde_lat = int(closing.shape[1] * .02)

    # define known sure background
    markers[0:border_top, :] = 100
    sure_fg[0:border_top, :] = 100

    markers[len(markers) - border_top:len(markers) - 1, :] = 100
    sure_fg[len(markers) - border_top:len(markers) - 1, :] = 100

    markers[:, 0:borde_lat] = 100
    sure_fg[:, 0:borde_lat] = 100
    markers[:, len(markers[0, :]) - borde_lat:len(markers[0, :]) - 1] = 100
    sure_fg[:, len(markers[0, :]) - borde_lat:len(markers[0, :]) - 1] = 100

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[sure_fg == 0] = 0

    markersres = cv2.watershed(image_resized, markers)

    markersres[markersres == 201] = 255
    markersres[markersres == 101] = 0
    markersres[markersres == -1] = 255

    mask_pred = cv2.resize(markersres.astype(np.uint8), (image.shape[1], image.shape[0]))

    # Get connected components
    ret, labels = cv2.connectedComponents(mask_pred)

    # Compute area of each connected component
    area = []
    for i, lab in enumerate(np.unique(labels)):
        area.append(mask_pred[labels == lab].size)

    # Sort indexes from length of areas
    idx = sorted(range(len(area)), key=lambda k: area[k])

    # Skip background index
    idx = [n for _, n in enumerate(idx) if np.sum(mask_pred[labels == n]) != 0]

    # Check if more than three connected components (background and two paintings)
    if ret > 2:
        for i in idx:
            # Remove smallest areas
            if area[i] < .1*area[idx[-1]]:
                mask_pred[labels == i] = 0

    # Remove indexes labels where its pixels are now zero
    idx = [n for _, n in enumerate(idx) if np.sum(mask_pred[labels == n]) != 0]

    if len(idx) > 3:
        mask_pred[labels == idx[0]] = 0
        idx.remove(idx[0])

    cv2.imwrite('masks_qd/mask_' + str(num_image) + '.png', mask_pred)

    # Remove indexes labels where its pixels are now zero
    idx = [n for _, n in enumerate(idx) if np.sum(mask_pred[labels == n]) != 0]

    obj = [np.where(labels == i) for i in idx]

    x_axis = [min(obj[1]) for obj in obj]
    obj_idx = sorted(range(len(x_axis)), key=lambda k: x_axis[k])

    mask = []

    for o, o_idx in enumerate(obj_idx):
        aux = np.zeros(mask_pred.shape[:], np.uint8)
        aux[obj[o_idx]] = 255

        mask.append(aux)

        #cv2.imwrite('masks_qd/mask_' + str(num_image) + '_' + str(o) +'.png', aux)

    return mask

def find_angle(idx, image, masks):

    rot_angle = []

    for num_mask, mask in enumerate(masks):

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        cropped, mask_cropped, x0, y0 = crop(image, mask)

        cropped_rs = resize_to_min(cropped, [500, 500])

        lines = estimate_lines(cropped_rs, (3, 10, 40))

        angles = np.asarray([[l, np.rad2deg(np.arctan2(line[1][-1] - line[0][-1], line[1][0] - line[0][0])),
                              np.sqrt(pow(line[1][-1]-line[0][-1], 2) + pow(line[1][0]-line[0][0], 2))]
                             for l, line in enumerate(lines)])

        angle = angles[abs(angles[:, 1]) < 45, :]

        if len(angle) > 1:
            ransac = RANSAC(max_trials=300, residual_threshold=5.0, random_state=0).fit(
                np.expand_dims(angle[:, 0], axis=1), angle[:, 1], angle[:, 2])
            inlier_mask = ransac.inlier_mask_

            # Predict data of estimated models
            line_X = np.arange(angle[:, 0].min(), angle[:, 0].max())[:, np.newaxis]
            line_y_ransac = ransac.predict(line_X)

            rot_angle.append(line_y_ransac[0])

    return rot_angle

def addPadding(images):

    padding=[]
    for image in images:
        [r,c] = [500,500]
        imPadding = cv2.copyMakeBorder(image, r , r, c, c, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padding.append(imPadding)

    return padding

def rotateImage(images, angles):

    rotated = []
    for image, angle in zip(images, angles):
        rgb = Image.fromarray(image) #src
        uint8image = np.asarray(rgb.rotate(angle), 'uint8')
        rotated.append(uint8image)
    return rotated

def hog_features(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (200, 200))

  hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8))
  hog_image = hog_image.ravel()

  return np.array(hog_image).flatten()

def agroup_paintings(distances,numPaintings):
    dist_matrix = []
    count = 0
    for num in numPaintings:
        dist_image = []
        for _ in range(num):
            dist_image.append(distances[count])
            count = count + 1
        dist_matrix.append(dist_image)

    return dist_matrix



def compute_MatrixRetrieval(distances, K):

    result = []
    for distance in distances:
        aux = []
        for dist in distance:
            matrix = (np.argsort(dist)[:K]).tolist()
            print(dist[matrix[0]])
            if dist[matrix[0]] > 57:
                matrix = [-1] + matrix[:9]
            aux.append(matrix)
        result.append(aux)

    return result

def evaluate(distances,numPaintings, K):
    dist_matrix = agroup_paintings(distances, numPaintings)
    result = compute_MatrixRetrieval(dist_matrix, K=K)

    pickle_out = open("result_qsd1_w5.pkl", "wb")
    pickle.dump(result, pickle_out)
    pickle_out.close()

    with open('./qsd1_w5/gt_corresps.pkl', mode='rb') as fd:
        gt = pickle.load(fd)

    gt2 = []
    for i in gt:
        dist_image = []
        for j in range(len(i)):
            dist_image.append([i[j]])
        gt2.append(dist_image)

    mapk = np.mean([metrics.average_precision.mapk(a, p, 3) for a, p in zip(gt2, result)])
    print('M@p{} is '.format(K, mapk * 100))

    return mapk

def clustering(features, cluster, num_rooms, dim_reduction):

    if dim_reduction:
        features = PCA(n_components=256,random_state=0).fit_transform(features)

    if cluster is 'KMeans':
        kmeans = KMeans(n_clusters=num_rooms, random_state=0).fit(features)
        clusters = kmeans.labels_

    elif cluster is 'GaussianMixture':
        gmm = GaussianMixture(n_components=num_rooms, covariance_type='full').fit(features)
        clusters = gmm.predict(features)

    elif cluster is 'BayesianGaussianMixture':
        dpgmm = BayesianGaussianMixture(n_components=num_rooms, covariance_type='full').fit(features)
        clusters = dpgmm.predict(features)

    return clusters

