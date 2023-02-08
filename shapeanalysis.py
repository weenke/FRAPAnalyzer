import numpy as np
import cv2 as cv
import operator


from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter
from scipy.spatial import distance
from skimage.exposure import rescale_intensity, equalize_adapthist


class ShapeAnalysis:

    def _image_preprocessing(self, return_image=False):
        
        img_historgram_correction = rescale_intensity(equalize_adapthist(self.image_array), out_range='uint8')
        #img_median_filtered = median_filter(img_historgram_correction, (10,10))
        img_bilating = cv.bilateralFilter(img_historgram_correction, 5, 40, 10)
        img_erosion = cv.erode(img_bilating, self.kernel, iterations=2)
        if return_image == False:
            return img_erosion
        else:    
            return img_erosion

    def _binary_gaussian_mixture(self):
        n_comp = 3
        labels = GaussianMixture(n_components=n_comp).fit_predict(self.processed_image.ravel().reshape(-1, 1))
        labels_to_image = np.array((labels.reshape(self.image_shape)), dtype='uint8')
        label_means = [(i, np.mean(self.image_array[np.where(labels_to_image==i)])) for i in range(n_comp)]
        label_means.sort(key=operator.itemgetter(1))

        labeled_image = np.zeros(self.image_shape)
        labeled_image[np.where(labels_to_image == label_means[0][0])] = 255

        labeled_image_uint8 = np.array(labeled_image, dtype='uint8')

        return labeled_image_uint8

    def _image_contours(self):
        contrours, heirarchy = cv.findContours(self.labeled_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #image_copy = self.image_array.copy()
        #cv.drawContours(image_copy, contrours, -1, 255, 1)
        return contrours

    def circle_fit_parameters(self):
        (x, y), radius = cv.minEnclosingCircle(self.biggest_area_contour())
        radius = int(radius)
        center = (int(x), int(y))
        return radius, center
    

    def __init__(self, image):
        self.image_array = np.array(image, dtype='uint8')
        self.image_shape = self.image_array.shape
        self.kernel = np.ones((3,3), np.uint8)
        self.processed_image = self._image_preprocessing()
        self.labeled_image = self._binary_gaussian_mixture()
        self.contours = self._image_contours()
        self.radius, self.center = self.circle_fit_parameters()
        
    

    def biggest_area_contour(self):
        return max(self.contours, key=lambda item: cv.contourArea(item))

    def image_ba_contour(self):
        image_copy = self.image_array.copy()
        cv.drawContours(image_copy, [self.biggest_area_contour()], -1, 255, 1)
        return image_copy

    def getCentroid(self):  #get centroid of the shape defined by the mainContour contour
        main_contour = self.biggest_area_contour()
        M = cv.moments(main_contour)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        
        return (cX, cY)

    def displayCenter(self):
        im_copy = self.image_array.copy()
        x, y = self.getCentroid()[0], self.getCentroid()[1]
        im_copy[:,int(x)] = 255
        im_copy[int(y),:] = 255
        return im_copy

    
    def apply_mask_circle_fit(self):
        radius = self.radius
        mask = np.zeros(self.image_shape)
        cv.circle(mask, self.getCentroid(), radius, 255, cv.FILLED)
        return mask

    def analysis_output(self):
        original = self.image_array
        mask = self.apply_mask_circle_fit()
        mask_center = self.getCentroid()
        
        return [original, mask, mask_center]

    def mean_background_intenisity_no_constrains(self):
        mean_value = np.mean(self.image_array[np.where(self.apply_mask_circle_fit() == 0)])
        return mean_value

    def mean_background_intenisity_full_frame(self):
        mean_value = np.mean(self.image_array)
        return mean_value

    def min_shape_intensity(self):
        min_intensity = np.min(self.image_array[np.where(self.apply_mask_circle_fit() == 255)])
        return min_intensity


    def roi_array_w_parameters(self):
        radius, center = self.radius, self.getCentroid()
        start = [self.getCentroid()[1]-radius, self.getCentroid()[1]+radius]
        fin = [self.getCentroid()[0]-radius, self.getCentroid()[0]+radius]
        roi = self.image_array[start[0]:start[1],fin[0]:fin[1]]
        return roi

    def draw_cricle_set_radius(self, set_radius):
        im_copy = self.image_array.copy()
        cv.circle(im_copy, self.getCentroid(), set_radius, 255, 1)
        return im_copy


#Outputs the data needed for NormalizedImageAnalysis class except for mean_infinite_backgroud_value, this one needs to be computed using the first frame\n
# and depending on the type of the FRAP experiment is extracted using mean_background_intenisity_no_constrains() or  mean_background_intenisity_no_constrains()
    def class_output(self):                                      
        roi_image = self.roi_array_w_parameters()
        shape_center = self.getCentroid()
        return [roi_image, shape_center]
        

class NormalizedImageAnalysis:
    
    def __init__(self, roi_data, normalization_data):
        self.roi_image_array = roi_data[0]
        self.mean_infinite_background_value = normalization_data
        self.roi_shape = self.roi_image_array.shape
        self.main_shape_center = (int(self.roi_shape[0]/2),int(self.roi_shape[1]/2))
        self.roi_min_intensity = np.min(self.roi_image_array)

    def imageNormalization(self):
        background_mean_value = self.mean_infinite_background_value    
        outside_intensity = np.ones(self.roi_shape)*background_mean_value
        #print(outside_intensity)
        minimal_instide_intensity = np.ones(self.roi_shape)*self.roi_min_intensity
        #print(minimal_instide_intensity)
        #eq_normalazation = lambda intensity: round(((intensity-minimal_instide_intensity)/(outside_intensity-minimal_instide_intensity)),2)
        #img_normalized_concentration = np.vectorize(eq_normalazation)
        #output = img_normalized_concentration(self.array)
        constant = np.subtract(outside_intensity, minimal_instide_intensity)
        variable = np.subtract(self.roi_image_array, minimal_instide_intensity)
        normalized_array = np.divide(variable, constant)
        filtered_normalized_array = median_filter(normalized_array, (3,3))
        return filtered_normalized_array

    def radius_HWL_08(self, given_background_mean=None):
        normal_image = self.imageNormalization()
        locations_08 = np.where(normal_image >= 0.8)
        locations_array = np.array([locations_08[0], locations_08[1]], dtype='uint8')
        min_value_radius_08 = min([distance.euclidean(position, [self.main_shape_center[0],self.main_shape_center[1]]) for position in locations_array.T])
        return min_value_radius_08

    def mask_roi(self):
        normal_image = self.imageNormalization()
        return_image = (normal_image > 0.8).astype(int)
        return return_image

    def draw_cricle_set_radius(self, set_radius):
        im_copy = self.roi_image_array.copy()
        cv.circle(im_copy, self.main_shape_center, set_radius, 255, 1)
        return im_copy

class IntensityRecoveryAnalysis(ShapeAnalysis):

    
    def __init__(self, image):
        super().__init__(image)

    def defined_roi_mean_intensity(self, set_radius):
        mask = np.zeros(self.image_shape)
        cv.circle(mask, self.getCentroid(), set_radius, 255, cv.FILLED)
        mean_value = np.mean(self.image_array[np.where(mask==255)])
        photobleacing_correction = np.mean(self.image_array[np.where(mask==0)])
        return [mean_value, photobleacing_correction]


class MinorShapeAnalysis:


    def _binary_gaussian_mixture_outter_shape(self):
        n_comp = 3
        labels = GaussianMixture(n_components=n_comp).fit_predict(self.processed_image.ravel().reshape(-1, 1))
        labels_to_image = np.array((labels.reshape(self.image_shape)), dtype='uint8')
        label_means = [(i, np.mean(self.image_array[np.where(labels_to_image==i)])) for i in range(n_comp)]
        label_means.sort(key=operator.itemgetter(1))

        labeled_image = np.zeros(self.image_shape)
        labeled_image[np.where(labels_to_image == label_means[-1][0])] = 255

        labeled_image_uint8 = np.array(labeled_image, dtype='uint8')

        return labeled_image_uint8, label_means[-1][1]

    def _binary_gaussian_mixture_image(self):
        n_comp = 5
        labels = GaussianMixture(n_components=n_comp).fit_predict(self.processed_image.ravel().reshape(-1, 1))
        labels_to_image = np.array((labels.reshape(self.image_shape)), dtype='uint8')
        return labels_to_image

    def _image_preprocessing(self, return_image=False):
        
        #img_historgram_correction = rescale_intensity(equalize_adapthist(self.image_array), out_range='uint8')
        #img_median_filtered = median_filter(img_historgram_correction, (10,10))
        img_bilating = cv.bilateralFilter(self.image_array, 5, 40, 10)
        img_erosion = cv.erode(img_bilating, self.kernel, iterations=2)
        if return_image == False:
            return img_erosion
        else:    
            return img_erosion


    def __init__(self, image):
        self.image_array = np.array(image, dtype='uint8')
        self.image_shape = self.image_array.shape
        self.kernel = np.ones((3,3), np.uint8)
        self.processed_image = self._image_preprocessing()
        self.labeled_image, self.minor_shape_mean_intensity = self._binary_gaussian_mixture_outter_shape()

    def fill_with_minor_shape_mean(self):
        im_copy = self.image_array.copy()
        im_copy[self.labeled_image == 0] = self.minor_shape_mean_intensity
        return im_copy

    def contour_on_labeled_image(self):
        contours, heirarchy = cv.findContours(self.labeled_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return [contours, heirarchy]

    def contained_cnoturs(self):
        contained_contour_list = []
        contour_data = self.contour_on_labeled_image()
        for index, h_list in enumerate(contour_data[1][0]):
            if h_list[-1] != -1:
                contained_contour_list.append(contour_data[0][index])
        return contained_contour_list

    def max_area_contained_contour(self):
        contained_contour_list = []
        contour_data = self.contour_on_labeled_image()
        for index, h_list in enumerate(contour_data[1][0]):
            if h_list[-1] != -1:
                contained_contour_list.append([[contour_data[0][index]], h_list[-1]])
        max_contained_contour = max((contained_contour_list), key=lambda item: cv.contourArea(item[0][0]))
        return max_contained_contour

    def containing_contour_biggest(self):
        contained_contours = self.contained_cnoturs()
        contour_data = self.contour_on_labeled_image()
        for index, h_list in enumerate(contour_data[1][0]):
            if h_list[-1] != -1:
                parent_contour = h_list[-1]
        return parent_contour

    def containig_contours(self):
        parent_contour = None
        contour_data = self.contour_on_labeled_image()
        for index, h_list in enumerate(contour_data[1][0]):
            if h_list[-1] != -1:
                parent_contour = h_list[-1]
        return parent_contour

    def draw_contour_index(self, index):
        image_copy = self.image_array.copy()
        contours = self.contour_on_labeled_image()[0]
        cv.drawContours(image_copy, [contours[index]], -1, 255, 1)
        return image_copy


    def get_biggest_contour(self, conturs_list):
        return [max(conturs_list, key=lambda item: cv.contourArea(item))]

    def draw_contours(self, contours=None):
        image_copy = self.image_array.copy()
        if contours is None:
            contours = self.contour_on_labeled_image()[0]
            cv.drawContours(image_copy, contours, -1, 255, 1)
        else:
            cv.drawContours(image_copy, contours, -1, 255, 1)
        return image_copy

    def contour_mean_by_index(self, index):
        image_copy = self.image_array.copy()
        mask = np.zeros(self.image_shape)
        contour = self.contour_on_labeled_image()[0][index]
        cv.drawContours(mask, [contour], -1, 255, cv.FILLED)
        return np.mean(image_copy[np.where(mask == 255)])
        

    def mean_contour_intensity(self, contour):
        mask = np.zeros(self.image_shape)
        mask = cv.drawContours(mask, [contour], -1, 255, cv.FILLED)
        mean_contour_intensity = np.mean(self.image_array[np.where(mask==255)])
        return mean_contour_intensity

    def mean_contour_intensity_index(self, index):
        mask = np.zeros(self.image_shape)
        contours = self.contour_on_labeled_image()[0]
        mask = cv.drawContours(mask, contours[index], -1, 255, cv.FILLED)
        mean_contour_intensity = np.mean(self.image_array[np.where(mask==255)])
        return mean_contour_intensity

    def contour_mean_vlaue(self):
        contour_intensity_list = [(index, self.mean_contour_intensity(contour)) for index, contour in enumerate(self.contour_on_labeled_image()[0])]
        return contour_intensity_list
    
    def correctly_masked_image(self):
        contour_order = self.contour_on_labeled_image()[1]

    def draw_biggest_inner_contour(self):
        image_copy = self.image_array.copy()
        cv.drawContours(image_copy ,self.get_biggest_contour(self.contained_cnoturs()), -1, 255, 1)
        return image_copy

    def complement_mask(self):
        mask = np.zeros(self.image_shape)
        [minor_shape_contour, outter_shape_index] = self.max_area_contained_contour()
        main_shape_contour = self.contour_on_labeled_image()[0][outter_shape_index]
        cv.drawContours(mask, [main_shape_contour], -1, 255, cv.FILLED)
        cv.drawContours(mask, minor_shape_contour, -1, 0, cv.FILLED)
        return mask

    def main_shape_mean_photobleaching(self):
        mask = self.complement_mask()
        infinite_mean_value = np.mean(self.image_array[np.where(mask==255)])
        return int(infinite_mean_value)

    #def main_shape_index_init(self):  #use one second frame to get index of first shape

    def getCentroid(self):  #get centroid of the shape defined by the mainContour contour
        main_contour = self.max_area_contained_contour()[0][0]
        M = cv.moments(main_contour)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        
        return (cX, cY)

    def circle_fit_parameters(self):
        (x, y), radius = cv.minEnclosingCircle(self.max_area_contained_contour()[0][0])
        radius = int(radius)
        center = (int(x), int(y))
        return (radius, center)

    def roi_array_w_parameters(self):
        radius, center = self.circle_fit_parameters()[0], self.getCentroid()
        start = [self.getCentroid()[1]-radius, self.getCentroid()[1]+radius]
        fin = [self.getCentroid()[0]-radius, self.getCentroid()[0]+radius]
        roi = self.image_array[start[0]:start[1],fin[0]:fin[1]]
        return [roi, center]

    
class IntensityRecoveryAnalysis_v2(MinorShapeAnalysis):

    
    def __init__(self, image):
        super().__init__(image)

    def defined_roi_mean_intensity(self, set_radius):
        mask = np.zeros(self.image_shape)
        cv.circle(mask, self.getCentroid(), set_radius, 255, cv.FILLED)
        mean_value = np.mean(self.image_array[np.where(mask==255)])
        complement_mask = self.complement_mask()
        photobleacing_correction = np.mean(self.image_array[np.where(complement_mask==255)])
        return [mean_value, photobleacing_correction]

#def analysis_worker():