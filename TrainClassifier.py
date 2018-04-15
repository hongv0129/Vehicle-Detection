import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.ndimage.measurements import label as scipy_label 

from VehicleDetection import    extract_features, \
                                extract_single_image_features, \
                                display_images_var1, \
                                convert_color, \
                                get_hog_features, \
                                color_hist, \
                                bin_spatial, \
                                find_cars, \
                                add_weight_to_heatmap, \
                                apply_threshold, \
                                draw_boxes, \
                                draw_labeled_bboxes
import time

#"""
#==============================================================================
print("================================================")
print("Load Data Samples : ")

#Get File Paths - Method A
import os

CarDataPaths = []

basedir = './dataset/vehicles/'
folder_names = os.listdir(basedir)

for foldername in folder_names:
    folder_base = basedir + foldername + '/'
    file_names = os.listdir(folder_base)
    
    for filename in file_names:
        if(filename != '.DS_Store'):
            CarDataPaths.append(folder_base+filename)
            #print( folder_base+filename )
            
print('Number of Vehicle Images:', len(CarDataPaths))

with open('ListCars.txt', 'w') as file:
    for carpath in CarDataPaths:
        file.write(carpath+'\n')

#------------------------------------------------------------------------------
#Get File Paths - Method B
import os
import glob

NonCarDataPaths = []

basedir = './dataset/non-vehicles/'
folder_names = os.listdir(basedir)

for foldername in folder_names:
    NonCarDataPaths.extend(glob.glob(basedir+foldername+'/*'))
    #print( basedir+filename+'/*' )
    #print(glob.glob(basedir+filename+'/*'))
    
print('Number of Non-vehicle Images:', len(NonCarDataPaths))

with open('ListNonCars.txt', 'w') as file:
    for carpath in NonCarDataPaths:
        file.write(carpath+'\n')
        
#==============================================================================
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


print("================================================")
print("Demo of Feature Extraction: ")

COLOR_SPACE = 'YCrCb'
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32

car_ind = np.random.randint(0, len(CarDataPaths))
noncar_ind = np.random.randint(0, len(NonCarDataPaths))

car_img = mpimg.imread(CarDataPaths[car_ind])
noncar_img = mpimg.imread(NonCarDataPaths[noncar_ind])

images = [car_img, noncar_img]
titles = ["Car", "Non-Car"]
plt_fig = plt.figure(figsize=(5.4, 3), dpi=100)
display_images_var1(plt_fig, 1, 2, images, titles)

#--------------------------------------------------------------
#Virtualize Hog Channel 0:

car_feature, car_hog_image, car_feat_img= \
extract_single_image_features(car_img, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=0,
                                spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis=True)
                                
noncar_feature, noncar_hog_image, noncar_feat_img = \
extract_single_image_features(noncar_img, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=0,
                                spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis=True)


print("car_features shape: ", np.array(car_feature).shape)
print("noncar_features shape: ", np.array(noncar_feature).shape)
print("================================================")

images = [car_feat_img, car_hog_image, noncar_feat_img, noncar_hog_image]
titles = ["Car Ch0", "Car Ch0 HOG", "Non-Car Ch0", "Non-Car Ch0 HOG"]
plt_fig = plt.figure(figsize=(10.8, 3), dpi=100)
display_images_var1(plt_fig, 1, 4, images, titles, cmap='gray')

#--------------------------------------------------------------
#Virtualize Hog Channel 1:

car_feature, car_hog_image, car_feat_img= \
extract_single_image_features(car_img, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=1,
                                spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis=True)
                                
noncar_feature, noncar_hog_image, noncar_feat_img = \
extract_single_image_features(noncar_img, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=1,
                                spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis=True)


print("car_features shape: ", np.array(car_feature).shape)
print("noncar_features shape: ", np.array(noncar_feature).shape)
print("================================================")

images = [car_feat_img, car_hog_image, noncar_feat_img, noncar_hog_image]
titles = ["Car Ch1", "Car Ch1 HOG", "Non-Car Ch1", "Non-Car Ch1 HOG"]
plt_fig = plt.figure(figsize=(10.8, 3), dpi=100)
display_images_var1(plt_fig, 1, 4, images, titles, cmap='gray')

#--------------------------------------------------------------
#Virtualize Hog Channel 2:

car_feature, car_hog_image, car_feat_img= \
extract_single_image_features(car_img, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=2,
                                spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis=True)
                                
noncar_feature, noncar_hog_image, noncar_feat_img = \
extract_single_image_features(noncar_img, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=2,
                                spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis=True)


print("car_features shape: ", np.array(car_feature).shape)
print("noncar_features shape: ", np.array(noncar_feature).shape)
print("================================================")

images = [car_feat_img, car_hog_image, noncar_feat_img, noncar_hog_image]
titles = ["Car Ch2", "Car Ch2 HOG", "Non-Car Ch2", "Non-Car Ch2 HOG"]
plt_fig = plt.figure(figsize=(10.8, 3), dpi=100)
display_images_var1(plt_fig, 1, 4, images, titles, cmap='gray')

plt.show()

#==============================================================================
print("================================================")
print("SVM Training on Entire Dataset : ")

COLOR_SPACE = 'YCrCb'
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32

car_imgs = np.array(CarDataPaths)
noncar_imgs = np.array(NonCarDataPaths)

timestamp_start = time.time()

car_features = extract_features(car_imgs, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
                                hog_channel=HOG_CHANNEL, 
                                spatial_feat=True, hist_feat=True, hog_feat=True)

noncar_features = extract_features(noncar_imgs, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS, hist_range=(0, 256), orient=ORIENT, 
                                pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
                                hog_channel=HOG_CHANNEL,
                                spatial_feat=True, hist_feat=True, hog_feat=True)
                                
timestamp_end = time.time()

print( "Feature Extraction Time: ", round( (timestamp_end-timestamp_start), 2), "sec" )

combine = (car_features, noncar_features)
print("car_features shape: ", np.array(car_features).shape)
print("noncar_features shape: ", np.array(noncar_features).shape)
print("(car_features, noncar_features).shape == ", np.array(combine).shape)

X = np.vstack((car_features, noncar_features)).astype(np.float64)
#X = np.hstack((car_features, noncar_features)).astype(np.float64)
print("X.shape == ", X.shape)

#------------------------------------------------------------------------------
X_Scaler = StandardScaler().fit(X)
scaled_X = X_Scaler.transform(X)
print("scaled_X.shape == ", scaled_X.shape)

y = np.hstack(( np.ones(len(car_features)), np.zeros(len(noncar_features)) ))
print("y.shape == ", y.shape)

random_int = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.10, random_state=random_int)

timestamp_start = time.time()

SVC = LinearSVC()
SVC.fit(X_train, y_train)

timestamp_end = time.time()
print( "Training Time: ", round( (timestamp_end-timestamp_start), 2), "sec" )

accuracy = SVC.score(X_test, y_test)

print("Accuracy of SVC is ", round(accuracy, 4))

#==============================================================================
print("================================================")
print("Save SVM model with its parameters into a pickle file... ")
import pickle

# save it to disk
#with open("svc_pretrained.p", "wb" ) as file:
#    pickle.dump(SVC, file)

# save the classifier model and its parameters to a file
with open("svc_pretrained.p", "wb" ) as file:
    dist_pickle = {}
    dist_pickle["svc"] = SVC
    dist_pickle["scaler"] = X_Scaler
    dist_pickle["orient"] = ORIENT
    dist_pickle["pix_per_cell"] = PIX_PER_CELL
    dist_pickle["cell_per_block"] = CELL_PER_BLOCK
    dist_pickle["spatial_size"] = SPATIAL_SIZE
    dist_pickle["hist_bins"] = HIST_BINS
    dist_pickle["color_space"] = COLOR_SPACE
    dist_pickle["hog_channel"] = HOG_CHANNEL
    #dist_pickle["X_test"] = X_test
    #dist_pickle["y_test"] = y_test
    pickle.dump(dist_pickle, file)

print("Done. ")
#"""

#==============================================================================
print("================================================")
print("Verify SVM classifier on actual images recorded by car front-view camera...")
import glob
import pickle

test_image_folder = './test_images/*'
test_image_paths = glob.glob(test_image_folder)

list_images = []
list_img_box_groups = []
list_heatmaps = []

list_display_images = []
image_titles = []

# load the classifier later
#with open("svc_pretrained.p", "rb" ) as file:
#    SVC = pickle.load(file)

# read the classifier model and its parameters from the file
with open("svc_pretrained.p", "rb" ) as file:
    dist_pickle = pickle.load(file)
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"] 
    hog_channel = dist_pickle["hog_channel"]
    
# Retrieve classifier parameters from pickle file above
# instead of hardcoded them below

#svc = SVC
#X_scaler = X_Scaler 
#orient = 9
#pix_per_cell = 8
#cell_per_block = 2
#spatial_size = (32, 32)
#hist_bins = 32

scale = 1.5
img_count = 0

for image_path in test_image_paths:
    
    timestamp_start = time.time()

    img = mpimg.imread(image_path)
    if image_path[-4:] == ".png":
        img = np.uint8(img*255)
    #img_tmp = cv2.imread(image_path)
    #img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        
    img_count = img_count + 1
    
    ystart = 380 
    ystop = 636

    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255.0
    
    # Make a heatmap of zeros
    heatmap = np.zeros_like(img[:,:,0])
    img_boxes = []
    
    img_tosearch = img[ystart: ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    #ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255.0
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    window_count = 0
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            
            window_count = window_count + 1
            
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #print("Test Feature Shape: ", spatial_features.shape, hist_features.shape, hog_features.shape)
            
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart), (0,0,255), 6) 
                
                img_boxes.append((  (xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)     ))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    timestamp_end = time.time()
    
    print("Image " + str(img_count).zfill(2) + ": " + str(round( (timestamp_end-timestamp_start), 2)) + " sec to process " + str(window_count) + " windows - " + str(len(img_boxes)) + " marked" )
    
    list_images.append(draw_img)
    list_img_box_groups.append(img_boxes)
    list_heatmaps.append(heatmap)

    list_display_images.append(draw_img)
    print(draw_img.shape)
    image_titles.append(" ")
    #image_titles.append("Test Image " + str(img_count).zfill(2))
    mpimg.imsave("output_images_SingleScale/" + str(img_count).zfill(2) + "_1_" + ".jpg", draw_img) 
    
    list_display_images.append(heatmap)
    image_titles.append(" ")
    #image_titles.append("Heat Map " + str(img_count).zfill(2))
    mpimg.imsave("output_images_SingleScale/" + str(img_count).zfill(2) + "_2_" + ".jpg", heatmap) 
    
#==============================================================================

plt_fig = plt.figure(figsize=(10.8, 32), dpi=300)

display_images_var1(plt_fig, 6, 2, list_display_images, image_titles)  #len(list_display_images)
#plt.show()

#==============================================================================
print("================================================")
print("Verify SVM classifier on actual images - now apply multiple sliding window scales: ")

img_count = 0
boShowAllBoxes = 1

for image_path in test_image_paths:
    
    timestamp_start = time.time()

    img = mpimg.imread(image_path)
    if image_path[-4:] == ".png":
        img = np.uint8(img*255)    
    #img_tmp = cv2.imread(image_path)
    #img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
    
    img_count = img_count + 1
    
    rectangle_boxes = []
    
    searched_window_count = 0
    
    #------------------------------------------------------
    # find cars in 192 x 192 sliding windows (scale=3) 
    # searching area: (ystart - ystop) = 636 - 396 = 240 = 192 x 1.25 (Chosen)
    #_, img_boxes, _ = \
    #find_cars(  img, conv_color='RGB2YCrCb', ystart=396, ystop=636, scale=3, svc=svc, 
    #            X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
    #            spatial_size=spatial_size, hist_bins=hist_bins )
    #
    #rectangle_boxes.extend(img_boxes)

    #------------------------------------------------------
    # find cars in 160 x 160 sliding windows (scale=2.5) 
    # searching area: (ystart - ystop) = 620 - 380 = 240 = 160 x 1.5 (Chosen)
    # searching area: (ystart - ystop) = 630 - 430 = 200 = 160 x 1.25
    _, img_boxes, _, searched_windows = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=380, ystop=620, scale=2.5, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
    
    rectangle_boxes.extend(img_boxes)
    
    searched_window_count += len(searched_windows)
    all_windows_img = draw_boxes(img, searched_windows, color=(200, 200, 0), thick=6)
    
    #------------------------------------------------------
    # find cars in 128 x 128 sliding windows (scale=2) 
    # searching area: (ystart - ystop) = 636 - 380 = 256 = 128 x 2
    # searching area: (ystart - ystop) = 572 - 380 = 192 = 128 x 1.5  (Chosen)
    # searching area: (ystart - ystop) = 540 - 380 = 160 = 128 x 1.25
    _, img_boxes, _, searched_windows = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=380, ystop=540, scale=2.0, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
    
    rectangle_boxes.extend(img_boxes)
    
    searched_window_count += len(searched_windows)
    all_windows_img = draw_boxes(all_windows_img, searched_windows, color=(200, 0, 0), thick=5)
    
    #------------------------------------------------------
    # find cars in 112 x 112 sliding windows (scale=1.75)
    # searching area: (ystart - ystop) = 604 - 380 = 224 = 112 x 2 (Chosen)
    # searching area: (ystart - ystop) = 508 - 368 = 140 = 112 x 1.25 (Chosen)    
    _, img_boxes, _, searched_windows = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=368, ystop=508, scale=1.75, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
                
    rectangle_boxes.extend(img_boxes)
    
    searched_window_count += len(searched_windows)
    all_windows_img = draw_boxes(all_windows_img, searched_windows, color=(200, 0, 200), thick=4)
    
    #------------------------------------------------------
    # find cars in 96 x 96 sliding windows (scale=1.5)
    # searching area: (ystart - ystop) = 572 - 380 = 192 = 96 x 2
    # searching area: (ystart - ystop) = 524 - 380 = 144 = 96 x 1.5
    # searching area: (ystart - ystop) = 488 - 368 = 120 = 96 x 1.25 (Chosen)
    # searching area: (ystart - ystop) = 464 - 368 = 96 = 96 x 1.0
    _, img_boxes, _, searched_windows = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=380, ystop=524, scale=1.5, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
                
    rectangle_boxes.extend(img_boxes)

    searched_window_count += len(searched_windows)
    all_windows_img = draw_boxes(all_windows_img, searched_windows, color=(0, 0, 200), thick=3)
    
    #------------------------------------------------------
    # find cars in 64 x 64 sliding windows (scale=1)
    # searching area: (ystart - ystop) = 508 - 380 = 128 = 64 x 2
    # searching area: (ystart - ystop) = 496 - 400 = 96 = 64 x 1.5 (Chosen)
    # searching area: (ystart - ystop) = 460 - 380 = 80 = 64 x 1.25
    _, img_boxes, _, searched_windows = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=400, ystop=496, scale=1.0, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
    
    rectangle_boxes.extend(img_boxes)
    
    searched_window_count += len(searched_windows)
    all_windows_img = draw_boxes(all_windows_img, searched_windows, color=(0, 200, 0), thick=2)
    
    #-------------------------------------------------------
    
    timestamp_end = time.time()
    
    print("Image " + str(img_count).zfill(2) + ": " + str(round( (timestamp_end-timestamp_start), 2)) + \
            " sec to process " + str(searched_window_count) + " windows >> detected " + str(len(rectangle_boxes)) + " hot windows" )
    
#------------------------------------------------------------------------------
    box_marked_img = draw_boxes(img, rectangle_boxes)
    
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = add_weight_to_heatmap(heatmap, rectangle_boxes)
    #heatmap  = apply_threshold(heatmap, 0)
    
    images = [box_marked_img, heatmap]
    titles = ["Marked Image", "Original Heat Map"]
    #plt_fig = plt.figure(figsize=(10.8, 3), dpi=200)
    #display_images_var1(plt_fig, 1, 2, images, titles)
    f, ax = plt.subplots(1, 2, figsize=(10.4, 3.15))
    f.tight_layout()
    ax[0].imshow(images[0])
    ax[0].set_title(titles[0], fontsize=14)
    ax[1].imshow(images[1], cmap='hot')
    ax[1].set_title(titles[1], fontsize=14)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)

    #-------------------------------------------------------
    heatmap  = apply_threshold(heatmap, 2)
    labels = scipy_label(heatmap)
    refined_marked_img = draw_labeled_bboxes(img, labels)
    
    images = [refined_marked_img, heatmap]
    titles = ["Marked Image", "Thresholded Heat Map"]
    #plt_fig = plt.figure(figsize=(10.8, 3), dpi=200)
    #display_images_var1(plt_fig, 1, 2, images, titles)
    f, ax = plt.subplots(1, 2, figsize=(10.4, 3.15))
    f.tight_layout()
    ax[0].imshow(images[0])
    ax[0].set_title(titles[0], fontsize=14)
    ax[1].imshow(images[1], cmap='hot')
    ax[1].set_title(titles[1], fontsize=14)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
    
    mpimg.imsave("output_images/image_demo_" + str(img_count).zfill(2) + "_0.jpg", refined_marked_img)
    #-------------------------------------------------------
    if(img_count == 4):
        images = [all_windows_img]
        titles = ["Searching windows"]
        #plt_fig = plt.figure(figsize=(10.8, 3), dpi=200)
        #display_images_var1(plt_fig, 1, 2, images, titles)
        f, ax = plt.subplots(1, 1, figsize=(12, 2))
        f.tight_layout()
        ax.imshow(images[0])
        ax.set_title(titles[0], fontsize=14)
        plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
        
        boShowAllBoxes = 0
        
plt.show()
    