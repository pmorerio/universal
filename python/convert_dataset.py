import numpy as np
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
from scipy.misc import imsave
import os


list_text_file = '/home/pmorerio/snap/skype/common/ucf101_singleFrame_RGB_test_split1.txt'

# put here the base folder for the dataset (no / at the end)
base_path = 'base_path'

# where to save, create it if does not exist
target_path = 'targetpath'
if not os.path.exists(target_path):
    os.mkdir(target_path)

# get universal perturbation from file
file_perturbation = os.path.join('data', 'universal.npy')
v = np.load(file_perturbation)


with open(list_text_file) as f:
    
    while True:

        line = f.readline()
        if not line:
            break

        video_folder = os.path.join(target_path,line.split('/')[0])
        if not os.path.exists(video_folder):
           os.mkdir(video_folder)
        
        path_test_image = os.path.join(base_path,line.split()[0])
        #print(path_test_image)
        
        # new filename (different parent folder)
        path_test_image_perturbed = os.path.join(target_path,line.split()[0])
        #print(path_test_image_perturbed)


        image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")

        # Clip the perturbation to make sure images fit in uint8
        clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)
        # Perturb image
        image_perturbed = image_original + clipped_v[None, :, :, :]

        # Save image in a new file

        undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8')

        imsave(path_test_image_perturbed, undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'))
