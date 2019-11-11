import pandas as pd
import pydicom as dcm
import glob , pylab
import os
import keras 
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import keras_preprocessing.image as KPImage
from PIL import Image

in_path = '/home/bif/Hello/stage_1_train_images/'
test_img_df = pd.DataFrame({'path': 
              glob.glob('/home/bif/Hello/stage_1_test_images/*.dcm')})
test_img_df['patientId'] = test_img_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

from keras.models import load_model
import numpy as np



img_gen_args = dict(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.05, 
                              width_shift_range = 0.02, 
                              rotation_range = 3, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range = 0.05,
                               )
img_gen = ImageDataGenerator(**img_gen_args)




def flow_from_dataframe(img_data_gen, in_df, seed = None, **dflow_args):
    base_dir = os.path.dirname(in_df['path'].values[0])
    print('## Ignore next message from keras, values are replaced anyways: seed: {}'.format(seed))
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                              seed = seed,
                                    **dflow_args)
    df_gen.filenames = in_df['path'].values
    df_gen.classes = np.stack(in_df['patientId'].values,0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

new_model = keras.models.load_model('full_model.h5')

IMG_SIZE = (224,224)
BATCH_SIZE = 24

def read_dicom_image(in_path):
    img_arr = dcm.read_file(in_path).pixel_array
    return img_arr/img_arr.max()
    
class medical_pil():
    @staticmethod
    def open(in_path):
        if '.dcm' in in_path:
            c_slice = read_dicom_image(in_path)
            int_slice =  (255*c_slice).clip(0, 255).astype(np.uint8) # 8bit images are more friendly
            return Image.fromarray(int_slice)
        else:
            return Image.open(in_path)
    fromarray = Image.fromarray
KPImage.pil_image = medical_pil


img = flow_from_dataframe(img_gen,test_img_df, 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE)

new_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


test_steps = 2*test_img_df.shape[0]//BATCH_SIZE
out_ids, out_vec = [], []
for _, (t_x, t_y) in zip(tqdm(range(test_steps)),img):
    out_vec += [new_model.predict(t_x)]
    out_ids += [t_y]
out_vec = np.concatenate(out_vec, 0)
out_ids = np.concatenate(out_ids, 0)
