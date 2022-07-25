%matplotlib qt5

from readlif.reader import LifFile, LifImage
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import csv
import os
from matplotlib import cm
from scipy.spatial import distance
import datatable as dt

class dataLoader:

    def _getRuns(self, args):
        
        aranged_runs = {}

        name_check = self.data[0].name.split(' ')[0].split('/')[0]
        images_in_one_run = []

        for i in self.data:
            current_img_name = i.name.split(' ')[0].split('/')[0]
            if current_img_name == name_check:
                images_in_one_run.append(i)
            else:
                aranged_runs[f'{name_check}'] = images_in_one_run
                name_check = i.name.split(' ')[0].split('/')[0]
                images_in_one_run = [i]
                
            aranged_runs[f'{name_check}'] = images_in_one_run
            
        return aranged_runs



    def __init__(self, lifrundict):
        self.data = lifrundict
        self.runs = self._getRuns(lifrundict)
        self.keys = list(self.runs.keys())

    def getRunByName(self, name):
        return self.runs[name]

    def getFrameInRun(self, name):

        run = self.getRunByName(name)

        list_of_frames = []
        for image in run:
            for frame in image.get_iter_t():
                list_of_frames.append(frame)

        return list_of_frames

    def getTimeline(self, name):
        time_steps_all = []
        last_digit = None
        for i in self.getRunByName(name):
            if last_digit == None:
                var1 = np.arange(0,i.dims.t+1)*(1/i.scale[3])
                time_steps_all.extend(var1.tolist())
                last_digit = time_steps_all[-1]
            else:
                var1 = np.arange(1, i.dims.t)*(1/i.scale[3])+last_digit
                time_steps_all.extend(var1.tolist())

        return time_steps_all
    
    def getBleachedFrames(self, name):
        index_bleached_frames = []
        for index, frame in enumerate(self.getFrameInRun(name)):
            img_array = np.array(frame)
            if img_array[128, 128] == 255:
                index_bleached_frames.append(index)
            else:
                pass
        return (index_bleached_frames[0], index_bleached_frames[-1])

    def getParametersAll(self, name):

        

        
        pass
        
        



class FRAPAnalysis():

    def __init__(self):
        pass
        
class MainShapeAnalyzer:

    kernel = np.ones((5,5), np.uint8)

    def __init__(self, image):
        kernel = np.ones((5,5), np.uint8)
        self.array = np.array(image)
        self.onesCopy = np.zeros_like(self.array)
        self.filter_erode = cv.erode(self.array, kernel, iterations=1)
        self.filter_bilat = cv.bilateralFilter(self.filter_erode, 9, 50, 50)
        _, self.binary = cv.threshold(self.filter_bilat, 60, 255, cv.THRESH_BINARY)
        self.contours, self.heirarchy = cv.findContours(self.binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    
    def mainContour(self):
        max_contour = max(self.contours, key=lambda item: cv.contourArea(item))
        return max_contour
    
    def disaplyContourImg(self, filled=True):
        im_copy = self.array.copy()
        if filled == True:
            cv.drawContours(im_copy, [self.mainContour()], -1, 255, cv.FILLED)
        else:
            cv.drawContours(im_copy, [self.mainContour()], -1, 255, 1)
        return im_copy
    
    def getCentroid(self):  #get centroid of the shape defined by the mainContour contour

        M = cv.moments(self.mainContour())
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        
        return (cX, cY)
    
    def displayMaskedImg(self):
        img_mask_b = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        img_mask_g = cv.bitwise_and(self.array, self.array, mask=img_mask_b)
        img_mask_g[self.coordinatesInverseMaskedArea()] = np.mean(self.array[self.coordiatesMaskedArea()])
        return img_mask_g

    def displayMaskedImgProper(self):
        img_mask_b = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        img_mask_g = cv.bitwise_and(self.array, self.array, mask=img_mask_b)
        return img_mask_g
    
    def coordiatesMaskedArea(self):
        img_mask_b = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        main_area_coordinates = np.where(img_mask_b == 255)
        return main_area_coordinates
    
    def coordinatesInverseMaskedArea(self):
        img_mask_b = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        main_area_coordinates = np.where(img_mask_b != 255)
        return main_area_coordinates

    def displayMainMask(self):
        img_mask_b = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        return img_mask_b

    


class SecondaryShapeAnalyzer:
    
    def __init__(self, image, primay_iamge, **kwargs):
        kernel = np.ones((5,5), np.uint8)
        self.array = np.array(image)
        self.onesCopy = np.ones_like(self.array)
        self.filter_erode = cv.erode(self.array, kernel, iterations=1)
        self.filter_bilat = cv.bilateralFilter(self.filter_erode, 9, 50, 50)
        _, self.binary = cv.threshold(self.filter_bilat, 75, 255, cv.THRESH_BINARY_INV)
        self.contours, self.heirarchy = cv.findContours(self.binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.main_mask = MainShapeAnalyzer(primay_iamge).displayMainMask()
        self.mininal_normalazation_value = kwargs.get('norm_min')
        self.mean_background_value = kwargs.get('norm_mean')

    def mainContour(self):
        max_contour = max(self.contours, key=lambda item: cv.contourArea(item))
        return max_contour

    def disaplyContourImg(self, filled=True):
        im_copy = self.array.copy()
        if filled == True:
            cv.drawContours(im_copy, [self.mainContour()], -1, 255, cv.FILLED)
        else:
            cv.drawContours(im_copy, [self.mainContour()], -1, 255, 1)
        return im_copy

    def getCentroid(self):  #get centroid of the shape defined by the mainContour contour
        M = cv.moments(self.mainContour())
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        return (cX, cY)
    
    def displayCenter(self):
        im_copy = self.array.copy()
        x, y = self.getCentroid()[0], self.getCentroid()[1]
        im_copy[:,int(x)] = 255
        im_copy[int(y),:] = 255
        return im_copy
    
    def coordinatesMaskArea(self):
        img_mask_b = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        main_area_coordinates = np.where(img_mask_b == 255)
        return main_area_coordinates

    def coordiantedComplimentMask(self):
        compliment_mask = self.main_mask
        compliment_mask[self.coordinatesMaskArea()] = 0
        return compliment_mask


    def intensityValueBackground(self):
        if self.mean_background_value == None:
            intensity_value = np.mean(self.array[np.where(self.coordiantedComplimentMask() == 255)])
        else:
            intensity_value = self.mean_background_value
        return round(intensity_value)
    
    def imageMaskInner(self):
        img_mask = cv.drawContours(self.onesCopy, [self.mainContour()], -1, 255, cv.FILLED)
        return img_mask

    def intensityValueMinInner(self):
        if self.mininal_normalazation_value == None:
            intensity_value = min(self.array[np.where(self.imageMaskInner() == 255)])
        else:
            intensity_value = self.mininal_normalazation_value
        return intensity_value

    def imageNormalization(self):
        outside_intensity = self.intensityValueBackground()
        minimal_instide_intensity = self.intensityValueMinInner()
        eq_normalazation = lambda intensity: round(((intensity-minimal_instide_intensity)/(outside_intensity-minimal_instide_intensity)),2)
        img_normalized_concentration = np.vectorize(eq_normalazation)
        output = img_normalized_concentration(self.array)
        return output

    def valueRadiusHW(self):
        normal_img = self.imageNormalization()
        val_075_loc = np.where(normal_img >= 0.8)
        locs_arrya = np.array([val_075_loc[0],val_075_loc[1]])
        value_radius = min(distance.euclidean(position, [self.getCentroid()[1],self.getCentroid()[0]]) for position in locs_arrya.T)
        return round(value_radius)
    
    def intensityAverageInner(self, radius):
        mask_known_radius = cv.circle(self.onesCopy, self.getCentroid(), radius, 255, cv.FILLED)
        intensity_mean_values = np.mean(self.imageNormalization()[np.where(mask_known_radius == 255)])
        return intensity_mean_values

    def parametersNormalazation(self):
        parameters_dict = {
            'min_norm' : self.intensityValueMinInner(),
            'mean_norm' : self.intensityValueBackground(),
            'roi_radius' : self.valueRadiusHW()
        }
        return parameters_dict
