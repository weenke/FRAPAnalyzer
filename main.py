from cgi import test
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

#main loading of the LIF data

lif_file_name = 'data_in/090522_FAF_SAS_FRAP.lif' #data_in\090522_FAF_SAS_FRAP.lif
lif_files = LifFile(f'./{lif_file_name}')
imf_list = [i for i in lif_files.get_iter_image()]

class dataLoader:

    def _getRuns(self, args):
        
        aranged_runs = {}

        #name_check = self.data[0].name.split(' ')[0].split('/')[0]
        name_check = self.data[0].name.split('/')[0]
        images_in_one_run = []

        for i in self.data:
            current_img_name = i.name.split('/')[0]
            if current_img_name == name_check:
                images_in_one_run.append(i)
            else:
                aranged_runs[f'{name_check}'] = images_in_one_run
                name_check = i.name.split('/')[0]
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
        if index_bleached_frames:
            results = (index_bleached_frames[0], index_bleached_frames[-1])
        else:
            results = (0, 0)
        return results

    def getParametersAll(self, name=None):
        if name == None:
            parameter_dict = {}
            for run_name in self.keys:
                if len(self.getRunByName(run_name)) <= 1:
                    continue
                else:
                    frame = self.getFrameInRun(run_name)[self.getBleachedFrames(run_name)[1]+1]
                    try:
                        parameter_dict[f'{run_name}'] = analysisInit(frame).params
                    except:
                        parameter_dict[f'{run_name}'] = 'Failed'
#                parameter_dict[f'{run_name}'] = None
                print(f'{run_name} ----> DONE')
            return parameter_dict
        else:
            if len(self.getRunByName(run_name)) <= 1:
                pass
            else:
                frame = self.getRunByName(name)[self.getBleachedFrames(name)[1]+1]
                params = analysisInit(frame).params
            return params

    def getMeanIntensity(self, parameters, name=None):
        if name is None:
            intensity_value_series = {}
            for run_name in self.keys:
                values_list = []
                for index, frame in enumerate(self.getFrameInRun(run_name)):
                    if index <= self.getBleachedFrames(run_name)[-1]:
                        values_list.append(0)
                    else:
                        try:
                            getter_init = FRAPAnalysis(frame, parameters[f'{run_name}'])
                            values_list.append(getter_init.intensityAverageInner())
                        except:
                            values_list.append(0)
                intensity_value_series[f'{run_name}'] = values_list
                print(f'{run_name} ---> DONE')
            return intensity_value_series
        else:
            values_list = []
            for index, frame in enumerate(self.getFrameInRun(name)):
                if index <= self.getBleachedFrames(name)[-1]:
                    values_list.append(0)
                else:
                    getter_init = FRAPAnalysis(frame, parameters[f'{name}'])
                    values_list.append(getter_init.intensityAverageInner())

            return values_list

class analysisInit:

    def __mainContourGetter(self, image):
        starting_contour_idx = mainShapeChooser(image).getMaxContour()
        if starting_contour_idx is None:
            starting_contour_idx = 0
        else:
            pass
        return starting_contour_idx


    def __init__(self, image):
        self.var1 = MainShapeAnalyzer(image)
        self.var1.setContourByIndex = self.__mainContourGetter(image)
        self.var2 = SecondaryShapeAnalyzer(self.var1.displayMaskedImg(), image)
        self.params = self.var2.parametersNormalazation()

    

class mainShapeChooser:
    
    def __init__(self, image):
        self.main_shape = MainShapeAnalyzer(image)
        self.init_iamge = image

    def detercCricles(self, image):
        img1_array = image
        detected_circles = cv.HoughCircles(img1_array, cv.HOUGH_GRADIENT, 2, 20)
        if detected_circles is None:
            return False
        else:
            return True

    def getMaxContour(self):
        contour_list = self.main_shape.ordered_contours
        returned_index = None
        for index, contour in enumerate(contour_list):
            #self.main_shape.setContour(index)
            self.main_shape.setContour = contour_list[index]
            var1 = self.main_shape.displayMaskedImg()
            var2 = SecondaryShapeAnalyzer(var1, self.init_iamge)
            if self.detercCricles(var2.array) == False:
                continue
            else:
                returned_index = index
                break

        return returned_index

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
        self.ordered_contours = sorted(self.contours, key=lambda item: cv.contourArea(item), reverse=True)
        self.num_ordered_contours = len(self.ordered_contours)
        self.__set_contour = 0

    
    def mainContour(self):
        
        given_contour_list = self.__set_contour

        if type(given_contour_list) == int:
            max_contour = max(self.contours, key=lambda item: cv.contourArea(item))
        else:
            if type(given_contour_list) == tuple:
                max_contour = max(given_contour_list)
            else:
                max_contour = given_contour_list
        return max_contour

    @property
    def setContour(self):
        return self.__set_contour
    @setContour.setter
    def setContour(self, contour):
        self.__set_contour = contour

    @property
    def setContourByIndex(self):
        return self.__set_contour
    
    @setContourByIndex.setter
    def setContourByIndex(self, contour_index):
        self.__set_contour = self.ordered_contours[contour_index]


    def listContours(self):
        return [cv.contourArea(controur) for controur in self.contours]

    def listContoursOrd(self):
        return [cv.contourArea(controur) for controur in self.ordered_contours]

    
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
        _, self.binary = cv.threshold(self.filter_bilat, 100, 255, cv.THRESH_BINARY_INV)
        self.contours, self.heirarchy = cv.findContours(self.binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.main_mask = MainShapeAnalyzer(primay_iamge).displayMainMask()
        self.primary_image = np.array(primay_iamge)
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
        return img_mask_b

    def coordiantedComplimentMask(self):
        compliment_mask = np.array(self.main_mask).copy()
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
        #print(outside_intensity)
        minimal_instide_intensity = self.intensityValueMinInner()
        #print(minimal_instide_intensity)
        #eq_normalazation = lambda intensity: round(((intensity-minimal_instide_intensity)/(outside_intensity-minimal_instide_intensity)),2)
        #img_normalized_concentration = np.vectorize(eq_normalazation)
        #output = img_normalized_concentration(self.array)
        constant = np.subtract(outside_intensity, minimal_instide_intensity)
        variable = np.subtract(self.primary_image, minimal_instide_intensity)
        return np.divide(variable, constant)

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
            'min_norm' : float(self.intensityValueMinInner()),
            'mean_norm' : float(self.intensityValueBackground()),
            'roi_radius' : float(self.valueRadiusHW())
        }
        return parameters_dict
        

class FRAPAnalysis(SecondaryShapeAnalyzer):

    def _loadFrame(self, frame):
        main_shape_init = MainShapeAnalyzer(frame).displayMaskedImg()
        inner_shape_init = SecondaryShapeAnalyzer(main_shape_init, frame)
        return inner_shape_init

    def _defineROI(self, frame):
        ones_arry = np.ones_like(frame)
        mask_known_radius = cv.circle(ones_arry, self.center, self.roi_radius, 255, cv.FILLED)
        return cv.bitwise_and(self.array, self.array, mask=mask_known_radius)


    def __init__(self, frame, parameters):
        self.array = np.array(frame)
        self.min = np.ones_like(self.array)*parameters['min_norm']
        self.mean = np.ones_like(self.array)*parameters['mean_norm']
        self.roi_radius = parameters['roi_radius']
        self.load_frame = self._loadFrame(frame)
        self.center = self.load_frame.getCentroid()
        self.roi_definition = self._defineROI(frame)


    def normalizationSimple(self):
        constant = np.subtract(self.mean, self.min)
        variable = np.subtract(self.array, self.min)
        return np.divide(variable, constant)
        


    def parametrizedNormal(self):

        eq_normalazation = lambda intensity: round(((intensity-self.min)/(self.mean-self.min)),2)
        img_normalized_concentration = np.vectorize(eq_normalazation)
        output = img_normalized_concentration(self.array)
        return output


    def intensityAverageInner(self):
        mask_known_radius = cv.circle(self.load_frame.onesCopy, self.center, self.roi_radius, 255, cv.FILLED)
        intensity_mean_values = np.mean(self.normalizationSimple()[np.where(mask_known_radius == 255)])
        return intensity_mean_values               

        
class helper_functions:

    def __init__(self, main, secondary):
        self.main = MainShapeAnalyzer(main)
        self.secondary = SecondaryShapeAnalyzer(self.main, main)


if __name__ == '__main__':
    print('hello')