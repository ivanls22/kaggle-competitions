import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import matplotlib.pylab as plt
import os
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers.experimental import preprocessing
from skimage.transform import resize
#from matplotlib import pyplot as plt

def detect_num_cols_to_shrink(list_of_num_cols, dataframe):
 
    convert_to_int8 = []
    convert_to_int16 = []
    convert_to_int32 = []
    convert_to_float16 = []
    convert_to_float32 = []
    
    for col in list_of_num_cols:
        
        if dataframe[col].dtype in ['int', 'int8', 'int32', 'int64']:
            describe_object = dataframe[col].describe()
            minimum = describe_object[3]
            maximum = describe_object[7]
            diff = abs(maximum - minimum)
            if diff < 255:
                convert_to_int8.append(col)
            elif diff < 65535:
                convert_to_int16.append(col)
            elif diff < 4294967295:
                convert_to_int32.append(col)   
                
        elif dataframe[col].dtype in ['float', 'float16', 'float32', 'float64']:
            describe_object = dataframe[col].describe()
            minimum = describe_object[3]
            maximum = describe_object[7]
            diff = abs(maximum - minimum)

            if diff < 65535:
                convert_to_float16.append(col)
            elif diff < 4294967295:
                convert_to_float32.append(col) 
        
    list_of_lists = []
    list_of_lists.append(convert_to_int8)
    list_of_lists.append(convert_to_int16)
    list_of_lists.append(convert_to_int32)
    list_of_lists.append(convert_to_float16)
    list_of_lists.append(convert_to_float32)
    
    return list_of_lists

train = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
print("shape of train_transaction: ", train.shape, "\n")
print(train.info())

c = (train.dtypes == 'object')
n = (train.dtypes != 'object')
cat_cols = list(c[c].index)
num_cols = list(n[n].index) 

print(cat_cols, "\n")
print("number categorical features: ", len(cat_cols), "\n\n")
print(num_cols, "\n")
print("number numerical features: ", len(num_cols))

num_cols_to_shrink_trans = detect_num_cols_to_shrink(num_cols, train)

convert_to_int8 = num_cols_to_shrink_trans[0]
convert_to_int16 = num_cols_to_shrink_trans[1]
convert_to_int32 = num_cols_to_shrink_trans[2]

convert_to_float16 = num_cols_to_shrink_trans[3]
convert_to_float32 = num_cols_to_shrink_trans[4]

print("convert_to_int8 :", convert_to_int8, "\n")
print("convert_to_int16 :", convert_to_int16, "\n")
print("convert_to_int32 :", convert_to_int32, "\n")

print("convert_to_float16 :", convert_to_float16, "\n")
print("convert_to_float32 :", convert_to_float32, "\n")

for col in convert_to_int8:
    train[col] = train[col].astype('int8')  

for col in convert_to_int16:
    train[col] = train[col].astype('int16')  
    
for col in convert_to_int32:
    train[col] = train[col].astype('int32') 

for col in convert_to_float16:
    train[col] = train[col].astype('float16')
    
for col in convert_to_float32:
    train[col] = train[col].astype('float32')

for i in cat_cols:
    train[i] = train[i].astype('category')
    
print(train.info(), "\n")

test = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")
label = train.loc[:, train.columns == 'cancer']
#train_no_label = train.loc[:, train.columns != 'cancer']

X_train = []
label_filtered = []
j=0
path = '/kaggle/input/rsna-breast-cancer-detection/train_images/'
positive_count=0
negative_count=0

all_images = train['image_id'].to_list()
for index,img in enumerate(all_images):
    print(j)
    #print(img)
    #print(label.iloc[j][0])
    if label.iloc[j][0] == 1 or negative_count < 800:
        train_by_image = train.loc[(train['image_id'] == img)]
        patient_by_img = train_by_image[['patient_id']].loc[index][0]
        if 'CC'== train_by_image[['view']].loc[index][0]:
            if 'R'== train_by_image[['laterality']].loc[index][0]:
                try:
                    image_file = dicom.dcmread(path + str(patient_by_img) + '/' + str(img)+'.dcm')
                    image_to_resize = (image_file.pixel_array)[:, ::-1]
                    image_resized = (resize(image_to_resize, (400, 280), anti_aliasing=True))
                    image_normalized = (image_resized - np.min(image_resized)) /(np.max(image_resized) - np.min(image_resized))
                    X_train.append(image_normalized)
                    label_filtered.append(label.iloc[j][0])
                except:
                    pass
            else:
                #axis[j].imshow(image_file.pixel_array)
                try:
                    image_file = dicom.dcmread(path + str(patient_by_img) + '/' + str(img)+'.dcm')
                    image_to_resize = (image_file.pixel_array)
                    image_resized = (resize(image_to_resize, (400, 280), anti_aliasing=True))
                    image_normalized = (image_resized - np.min(image_resized)) /(np.max(image_resized) - np.min(image_resized))
                    X_train.append(image_normalized)
                    label_filtered.append(label.iloc[j][0])
                except:
                    pass
    negative_count = len(label_filtered) - sum(label_filtered)
    positive_count = sum(label_filtered)
    j = j + 1


print(len(X_train))
print(len(label_filtered))
print(label.sum())
print(label.count())
