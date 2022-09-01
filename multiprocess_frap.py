from turtle import delay
from frap_analyzer import *
import json
import pickle
import cv2 as cv
from joblib import Parallel, delayed
from multiprocessing import Manager, managers

def singular_process(name, parameters):
    try:
        timeline = data1.getTimeline(name)
        intensity_reading = data1.getMeanIntensity(parameters, name)
    except:
        timeline = [0]
        intensity_reading = [0]

    
    # posix_name = name.reaplce(' ','_')

    # with open(f'data_out/{posix_name}.json',)

    shared_results_dict[name] = [timeline, intensity_reading]
    


lif_file_name = '/data_in/160322-CrACr.lif' #data_in\090522_FAF_SAS_FRAP.lif
lif_files = LifFile(f'./{lif_file_name}')
imf_list = [i for i in lif_files.get_iter_image()]

data1 = dataLoader(imf_list)
#img1 = np.array(data1.getFrameInRun('FRAP 009')[1])

with open('params_160322_CrACr.pickle', 'rb') as pickle_f:
        parameters = pickle.load(pickle_f)

fixed_names = [name for name in data1.keys if "FRAP" in name]

if __name__ == '__main__':
    
    manager = Manager()
    shared_results_dict = manager.dict()

    # for name in fixed_names:
    #     singular_process(name, parameters)
    Parallel(n_jobs=4, verbose=1, backend='loky')(
       delayed(singular_process)(name, parameters) for name in fixed_names
    )

    with open('results_160322_CrACr.json', 'w') as fp:
        json.dump(shared_results_dict.copy(), fp)
