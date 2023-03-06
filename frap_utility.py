import numpy as np
from shapeanalysis import *
from readlif.reader import LifFile, LifImage
import matplotlib.pyplot as plt
import cv2 as cv


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
        #print('i am here')
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
        adder = 0
        for i in self.getRunByName(name):

            if i.dims[3] == 1:
                adder = 1
                continue
            else:
                pass

            if last_digit is None:
                var1 = np.arange(0,i.dims.t+1+adder)*(1/i.scale[3])
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
            mid_point = int(img_array.shape[0]/2)
            if img_array[mid_point, mid_point] == 255:
                index_bleached_frames.append(index)
            else:
                pass
        if index_bleached_frames:
            results = (index_bleached_frames[0], index_bleached_frames[-1])
        else:
            results = (0, 0)
        return results


def make_list(file_name):
    lif_file_name = f'/data_in/{file_name}'
    lif_files = LifFile(f'./{lif_file_name}')
    return [i for i in lif_files.get_iter_image()]

def _get_timelime_run(runs_dict, name, bleach_frames):


    time_scale_data = []

    for run in runs_dict[name]:
        if isinstance(run.scale_n[4], float):
            time_scale_data.append((1/run.scale_n[4], run.dims_n[4]))
    
    init_construct = np.arange(0, bleach_frames, 1)*time_scale_data[0][0]
    #print(init_construct)

    def append_to_timeline(range, scale):
        timetime_to_append = np.arange(1, range+1, 1)*scale
        adder_array = init_construct[-1]
        new_construct = np.append(init_construct, timetime_to_append+adder_array)
        
        return new_construct

    for parameters in time_scale_data:

        init_construct = append_to_timeline(parameters[1], parameters[0])

    return init_construct

def _get_runs_dicts(lif_files_list):

    frap_experiment_dict = {}

    def append_to_dict(file, name):
        if name in frap_experiment_dict:
            list_addition = frap_experiment_dict[f'{name}']
            list_addition.append(file)
            frap_experiment_dict[f'{name}'] = list_addition
        else:
            frap_experiment_dict[f'{name}'] = [file]
            
            
    name_check = lif_files_list[0].name.split('/')[0]

    for file in lif_files_list:
        current_file_name = file.name.split('/')[0]
        if current_file_name == name_check:
            append_to_dict(file, name_check)
        else:
            append_to_dict(file, current_file_name)
            name_check = current_file_name
    
    return frap_experiment_dict 


if __name__ == '__main__':
    print('not as module')