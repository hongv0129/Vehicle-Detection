
import os

import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.ndimage.measurements import label as scipy_label 

from VehicleDetection import    extract_features, \
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

import pickle

from moviepy.editor import VideoFileClip

#==============================================================================
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boVerifySVCAccuracy = False 

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
    
    if( boVerifySVCAccuracy == True):
        X_test = dist_pickle["X_test"]
        y_test = dist_pickle["y_test"]
        accuracy = svc.score(X_test, y_test)
        print("Accuracy of SVC is ", round(accuracy, 4))

#==============================================================================
"""
def show_words_on_image()
    font = cv2.FONT_HERSHEY_DUPLEX
    
        text = 'Number of : ' + '{:04.2f}'.format(left_curverad/1000.0) + 'km'
        cv2.putText(processed_img, text, (30,70), font, fontScale=1.2, color=(190,255,190), thickness=2, lineType=cv2.LINE_AA)
        
        text = 'Number of Detected frames: ' + '{:04.2f}'.format(right_curverad/1000.0) + 'km'
        cv2.putText(processed_img, text, (30,110), font, fontScale=1.2, color=(190,255,190), thickness=2, lineType=cv2.LINE_AA)
"""
#==============================================================================
class BoundingBoxes:

    #----------------------------------------------------------------
    def __init__(self, num_prev_records=32, num_prev_frames=32):
        self.num_prev_records = num_prev_records
        self.num_prev_heatmaps = num_prev_frames
        self.boxes_record = []
        self.boxes = []
        self.heatmap_record = []
        
    #----------------------------------------------------------------
    def reset(self, num_prev_records=32, num_prev_frames=32):
        self.num_prev_records = num_prev_records
        self.num_prev_heatmaps = num_prev_frames
        self.boxes_record = []
        self.boxes = []
        self.heatmap_record = []
        
    #----------------------------------------------------------------
    def add_boxes(self, list_boxes):
    
        if len(list_boxes) > 0:
            self.boxes_record.append(list_boxes)
            
            if( len(self.boxes_record) > self.num_prev_records ):
                self.boxes_record = self.boxes_record[ (len(self.boxes_record)-self.num_prev_records) : ]
            #print(" Number of boxes added : ", len(list_boxes))
            
    #----------------------------------------------------------------
    def get_all_boxes(self):
        self.boxes = []
        if( len(self.boxes_record) > 0):
            for list_boxes in self.boxes_record:
                self.boxes.extend(list_boxes)
        else:
            self.boxes = None
            
        #print(len(self.boxes_record))  #for debug
        #print(" Total number of boxes kept: ", len(self.boxes))
        return self.boxes
        
    #----------------------------------------------------------------
    def get_bboxes_threshold(self):
    
        HeatValidRatio = 0.80
        
        return len(self.boxes_record) * HeatValidRatio
        #return 2 + np.around((len(self.boxes_record))*0.80) #*1.3333 #//12*13 #0.75
        
    #----------------------------------------------------------------
    def add_heatmap(self, list_boxes, img):
    
        if len(list_boxes) > 0:
        
            heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
            heatmap = add_weight_to_heatmap(heatmap, list_boxes)
            #plt.imshow(heatmap, cmap='hot')
            #plt.show()
            
            heatmap[heatmap >= 2] = 1
            self.heatmap_record.append(heatmap)
            #plt.imshow(heatmap, cmap='hot')
            #plt.show()
            
            if( len(self.heatmap_record) > self.num_prev_heatmaps ):
                self.heatmap_record = self.heatmap_record[ (len(self.heatmap_record)-self.num_prev_heatmaps) : ]
            
    #----------------------------------------------------------------
    def get_framecount_threshold(self):
    
        FrameCountValidRatio = 0.70
        
        heatmap_sum = sum(self.heatmap_record)
        heatmap_thresholds = np.zeros_like(heatmap_sum).astype(np.float)
        
        #current_threshold = 3 + np.around(len(self.heatmap_record)*0.7)
        #current_threshold = np.around(self.num_prev_records * 0.75)
        current_threshold = self.num_prev_heatmaps * FrameCountValidRatio #0.62
        
        #print("Current Frame Threshold: ", current_threshold)
        #print(len(self.heatmap_record))
        #plt.imshow(heatmap_sum, cmap='hot')
        
        heatmap_thresholds[(heatmap_sum >= current_threshold) ] = 1
        #plt.imshow(heatmap_thresholds, cmap='hot')
        #plt.show()
            
        return heatmap_thresholds
        
    #----------------------------------------------------------------
    def get_all_heatmap(self):
        return self.heatmap_record
        
#==============================================================================

boOutputRawBoxes = False
boDebugMode = False 
boFullVideo = True

BoxHandler = BoundingBoxes() #using default num_prev_records for initialization
FrameCount = 0

def Process_video_frame(img):
    global BoxHandler
    global FrameCount
    
    FrameCount = FrameCount + 1
    #print("Frame Count :", FrameCount)
    
    rectangle_boxes = []
    
    #------------------------------------------------------
    # find cars in 192 x 192 sliding windows (scale=3) 
    # searching area: (ystart - ystop) = 636 - 396 = 240 = 192 x 1.25 (Chosen)
    #_, img_boxes, _, _ = \
    #find_cars(  img, conv_color='RGB2YCrCb', ystart=396, ystop=636, scale=3, svc=svc, 
    #            X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
    #            spatial_size=spatial_size, hist_bins=hist_bins )
    #
    #rectangle_boxes.extend(img_boxes)

    # find cars in 160 x 160 sliding windows (scale=2.5) 
    # searching area: (ystart - ystop) = 620 - 380 = 240 = 160 x 1.5 (Chosen)
    # searching area: (ystart - ystop) = 630 - 430 = 200 = 160 x 1.25
    _, img_boxes, _, _ = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=380, ystop=620, scale=2.5, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
    
    rectangle_boxes.extend(img_boxes)
    
    # find cars in 128 x 128 sliding windows (scale=2) 
    # searching area: (ystart - ystop) = 636 - 380 = 256 = 128 x 2
    # searching area: (ystart - ystop) = 572 - 380 = 192 = 128 x 1.5  (Chosen)
    # searching area: (ystart - ystop) = 540 - 380 = 160 = 128 x 1.25
    _, img_boxes, _, _ = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=380, ystop=540, scale=2, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
    
    rectangle_boxes.extend(img_boxes)
    
    # find cars in 112 x 112 sliding windows (scale=1.75)
    # searching area: (ystart - ystop) = 604 - 380 = 224 = 112 x 2 (Chosen)
    # searching area: (ystart - ystop) = 508 - 368 = 140 = 112 x 1.25 (Chosen)    
    _, img_boxes, _, _ = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=368, ystop=508, scale=1.75, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
                
    rectangle_boxes.extend(img_boxes)
    
    # find cars in 96 x 96 sliding windows (scale=1.5)
    # searching area: (ystart - ystop) = 572 - 380 = 192 = 96 x 2
    # searching area: (ystart - ystop) = 524 - 380 = 144 = 96 x 1.5
    # searching area: (ystart - ystop) = 488 - 368 = 120 = 96 x 1.25 (Chosen)
    # searching area: (ystart - ystop) = 464 - 368 = 96 = 96 x 1.0
    _, img_boxes, _, _ = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=380, ystop=524, scale=1.5, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
                
    rectangle_boxes.extend(img_boxes)

    # find cars in 64 x 64 sliding windows (scale=1)
    # searching area: (ystart - ystop) = 508 - 380 = 128 = 64 x 2
    # searching area: (ystart - ystop) = 496 - 400 = 96 = 64 x 1.5 (Chosen)
    # searching area: (ystart - ystop) = 460 - 380 = 80 = 64 x 1.25
    _, img_boxes, _, _ = \
    find_cars(  img, conv_color='RGB2YCrCb', ystart=400, ystop=496, scale=1, svc=svc, 
                X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                spatial_size=spatial_size, hist_bins=hist_bins )
    
    rectangle_boxes.extend(img_boxes)
    
    #-------------------------------------------------------
    # Debug: Generate video with raw boxes drawn 
    if (boDebugMode == True) and (boOutputRawBoxes == True):
        box_marked_img = draw_boxes(img, rectangle_boxes)
        return box_marked_img
    #------------------------------------------------------
    BoxHandler.add_boxes(rectangle_boxes)
    BoxHandler.add_heatmap(rectangle_boxes, img)
    
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    
    heatmap = add_weight_to_heatmap(heatmap, BoxHandler.get_all_boxes() )
    
    heatmap = apply_threshold(heatmap, BoxHandler.get_bboxes_threshold() )
    
    heatmap_thresholds = BoxHandler.get_framecount_threshold()
    #plt.imshow(heatmap_thresholds, cmap='hot')
    #plt.show()
    
    heatmap[(heatmap_thresholds==0)] = 0
    
    #-------------------------------------------------------
    # Debug: Generate video with Heat Map + Raw Boxes 
    if boDebugMode == True:
        heatmap_show = heatmap
        
        zeroimage = np.zeros_like(img[:,:,0]).astype(np.float)
        
        heatmap_RGB = cv2.merge([heatmap_show*3, zeroimage, zeroimage])
        heatmap_RGB = heatmap_RGB.astype(np.uint8)
        #plt.imshow(heatmap_RGB)
        #plt.show()
        
        debug_image = draw_boxes(img, rectangle_boxes, color=(0, 0, 255), thick=2)
        debug_image = cv2.addWeighted(debug_image, 1, heatmap_RGB, 0.8, 0)
        
        labels = scipy_label(heatmap)
        debug_image = draw_labeled_bboxes(debug_image, labels, color=(255, 0, 0), thick=3)
        return debug_image
    #-------------------------------------------------------
    else:
        labels = scipy_label(heatmap)
        refined_marked_img = draw_labeled_bboxes(img, labels)
    
    labels = scipy_label(heatmap)
    refined_marked_img = draw_labeled_bboxes(img, labels)
    
    return refined_marked_img
    # return box_marked_img, heatmap_RGB, refined_marked_img
    
#==============================================================================
if not os.path.exists("test_videos_output"):
    os.makedirs("test_videos_output")
    
if boFullVideo == False:
    
    BoxHandler.reset()
    test_video_output = "./project_video_hard_section_1_output.mp4"
    clip1 = VideoFileClip("./project_video_hard_section_1.mp4")
    test_clip = clip1.fl_image(Process_video_frame)
    test_clip.write_videofile(test_video_output, audio=False)
    
    BoxHandler.reset()
    test_video_output = "./test_video_output.mp4"
    clip_test = VideoFileClip("./test_video.mp4")
    test_clip = clip_test.fl_image(Process_video_frame)
    test_clip.write_videofile(test_video_output, audio=False)
    
    BoxHandler.reset()    
    test_video_output = "./project_video_hard_section_0_output.mp4"
    clip1 = VideoFileClip("./project_video_hard_section_0.mp4")
    test_clip = clip1.fl_image(Process_video_frame)
    test_clip.write_videofile(test_video_output, audio=False)

    BoxHandler.reset()
    test_video_output = "./project_video_hard_section_2_output.mp4"
    clip2 = VideoFileClip("./project_video_hard_section_2.mp4")
    test_clip = clip2.fl_image(Process_video_frame)
    test_clip.write_videofile(test_video_output, audio=False)
    
else:

    BoxHandler.reset()
    test_video_output = "./project_video_output.mp4"
    clip_entire = VideoFileClip("./project_video.mp4")
    test_clip = clip_entire.fl_image(Process_video_frame)
    test_clip.write_videofile(test_video_output, audio=False)
    