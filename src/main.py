from utils import *

# DEFINING GLOBAL VARIABLES
descriptor = 'ORB'

if descriptor is 'SIFT':
    method = 'FLANN'
    resize_factor = 1 / 4
    MIN_MATCHES = 80

elif descriptor is 'ORB':
    method = 'BRUTE_FORCE'
    resize_factor = 1 / 4
    MIN_MATCHES = 20

elif descriptor is 'SURF':
    method = 'FLANN'
    resize_factor = 1 / 4
    MIN_MATCHES = 80

def main():

    print('Loading images...')
    qsd = load_imgs('./qst1_w5')
    bbdd = load_imgs('../bbdd')

    qsd_shapes = np.asarray([image.shape[:2] for image in tqdm(qsd)])
    bbdd_shapes = np.asarray([image.shape[:2] for image in tqdm(bbdd)])

    min_shape = [min(qsd_shapes[:, 0]), min(qsd_shapes[:, 1])]

    #PREPROCESSING
    print('\nApplying denoising to query dataset...')
    qsd_denoised = [remove_noise(image) for image in tqdm(qsd)]

    print('\nRize all images before computations...')
    qsd_rs = [resize_to_min(image, min_shape) for image in tqdm(qsd_denoised)]
    bbdd_rs = [resize_to_min(image, min_shape) for image in tqdm(bbdd)]

    #TASK 1: Remove background
    print('\nGenerating background masks ...')
    qsd_mask = [remove_bg(idx, image) for idx, image in tqdm(zip(range(30), qsd_rs))]

    print('\nFinding angle of rotation ...')
    rot_angle = [find_angle(idx,image, masks) for idx, image, masks in tqdm(zip(range(30), qsd_denoised, qsd_mask))]
    print(rot_angle)

    print('\nFinding corners...')
    corners = [detect_corners(idx, mask) for idx, mask in tqdm(zip(range(30), qsd_mask))]
    print(corners)

    print('\nSaving angles and corners coord into pickle...')
    frames = []
    for corner, angle in zip(corners, rot_angle):
        painting = []
        for c, a in zip(corner, angle):
            painting.append([a, c])
        frames.append(painting)

    pickle_out = open("frames.pkl", "wb")
    pickle.dump(frames, pickle_out)
    pickle_out.close()

    print("Cropping paintings...")
    try:
        os.mkdir('cropped_paintings')
    except:
        print("Directory cropped_paintings already exists")
    cropped = [crop_paintings(idx, image, masks, True) for idx, image, masks in tqdm(zip(range(30), qsd_denoised, qsd_mask))]
    cropped_paints = [item[0] for item in cropped]
    cropped_masks = [item[1] for item in cropped]

    print("Adding padding to paintings...")
    cropped_padding = [addPadding(image) for image in tqdm(cropped_paints)]
    cropped_padding_masks = [addPadding(image) for image in tqdm(cropped_masks)]

    print("Rotating paintings...")
    try:
        os.mkdir('rotated_cropped_paintings')
    except:
        print("Directory rotated_cropped_paintings already exists")
    rotated = [rotateImage(image, angle) for image, angle in tqdm(zip(cropped_padding, rot_angle))]
    rotated_masks = [rotateImage(image, angle) for image, angle in tqdm(zip(cropped_padding_masks, rot_angle))]

    print("Cropping rotated images...")
    rotated_cropped = [crop_paintings(idx, image, masks, False) for idx, image, masks in tqdm(zip(range(30), rotated, rotated_masks))]
    rotated_cropped_paints = [item[0] for item in rotated_cropped]

    print('\nDetecting points of interest and its descriptors...')
    flat_rotated = [rot for rotated_paint in rotated_cropped_paints for rot in rotated_paint]
    qsd_kp_des = [find_keypoints(image, descriptor) for image in tqdm(flat_rotated)]
    bbdd_kp_des = [find_keypoints(image, descriptor) for image in tqdm(bbdd_rs)]

    # TASK 2: ORB Descriptors
    print('\nFinding Matches between BBDD images and QSD1_w5 images ...')
    dist = [find_correspondences(bbdd_kp_des, qs_kp_des, method,10) for qs_kp_des in tqdm(qsd_kp_des)]

    # TASK 3: Evaluation
    print('\nPredicting results from distances and evaluating them ...')
    numPaintings = [len(mask) for mask in qsd_mask]
    mapk = evaluate(dist, numPaintings, 3)

    # TASK 4: Rooms clustering
    print('\nClustering paintings in 10 rooms ...')
    cluster = 'KMeans'
    num_rooms = 10
    bbdd_hog = np.asarray([hog_features(image) for image in tqdm(bbdd)])

    clusters = clustering(bbdd_hog, cluster, num_rooms, True)

    for num_room in range(num_rooms):
        name_room = 'room' + str(num_room)
        if not os.path.exists(name_room):
            os.mkdir(name_room)

    [cv2.imwrite('room'+str(room)+'/'+str(idx)+'.jpg', image) for idx, image, room in tqdm(zip(range(len(bbdd)), bbdd, clusters))]

main()
