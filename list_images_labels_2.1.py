from random import shuffle
import glob
import cv2
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Not segregated into train, val, and test

"""
Source: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
"""


"""
List images and their labels
"""
 # data order
data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

# shuffle the addresses before saving
shuffle_data = False

# Change teh label to one hot encoding
one_hot_encoding = True


# read addresses and labels from the folder
dme_path = './dataset/Cropped_BM3D/dme/*.png'
normal_path = './dataset/Cropped_BM3D/normal/*.png'

addrs_dme = glob.glob(dme_path)
labels_dme = [1 for addr in addrs_dme]  # 0 = Normal, 1 = DME

addrs_normal = glob.glob(normal_path)
labels_normal = [0 for addr in addrs_normal]  # 0 = Normal, 1 = DME

addrs = addrs_dme+addrs_normal
labels = labels_dme+labels_normal

# shuffle data
if shuffle_data:
    # address to where you want to save the hdf5 file
    hdf5_path = './dataset/Cropped_BM3D/Cropped_BM3D_4.hdf5'
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

else:
    # address to where you want to save the hdf5 file
    hdf5_path = './dataset/Cropped_BM3D/Cropped_BM3D_4.hdf5'



# one hot encoding
if one_hot_encoding:
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# check the order of data and chose proper data shape to save images
if data_order == 'th':
    shape = (len(addrs), 3, 224, 224)
elif data_order == 'tf':
    shape = (len(addrs), 224, 224, 3)

print('shape', shape)

# open a hdf5 file and create arrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("img", shape, np.int8)
hdf5_file.create_dataset("labels",  (len(addrs),), np.int8)
hdf5_file.create_dataset("labels_OHE",  (len(addrs), 2), np.int8)


# loop over addresses
for i in range(len(addrs)):
    # print how many images are saved every 1000 images

    if i % 1000 == 0 and i > 1:
        print('Data: {}/{}'.format(i, len(addrs)))


    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # add any image pre-processing here

    """
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    """
    # save the image and calculate the mean so far
    hdf5_file["img"][i, ...] = img[None]

hdf5_file["labels"][...] = labels
hdf5_file["labels_OHE"][...] = onehot_encoded

# save the mean and close the hdf5 file
hdf5_file.close()

print("Complete...")


"""
# Read the HDF5 file


import h5py
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

hdf5_path = 'C:/Users/User/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/Cropped_BM3D_2.hdf5'
subtract_mean = False

# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

# subtract the training mean
if subtract_mean:
    mm = hdf5_file["mean"][0, ...]
    mm = mm[np.newaxis, ...]

# Total number of samples
data_num = hdf5_file["img"].shape[0]

# create list of batches to shuffle the data
nb_class = 2
batch_size = 5
batches_list = list(range(int(ceil(float(data_num) / batch_size))))
shuffle(batches_list)

# loop over batches
for n, i in enumerate(batches_list):
    i_s = i * batch_size  # index of the first image in this batch
    i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch

    # read batch images and remove training mean
    images = hdf5_file["img"][i_s:i_e, ...]
    if subtract_mean:
        images -= mm

    # read labels and convert to one hot encoding
    labels = hdf5_file["labels"][i_s:i_e]
    labels_one_hot = np.zeros((batch_size, nb_class))
    labels_one_hot[np.arange(batch_size), labels] = 1

    print(n+1, '/', len(batches_list))
    print(labels[0], labels_one_hot[0, :])

    plt.imshow(images[0])
    plt.show()
    if n == 5:  # break after 5 batches
        break
hdf5_file.close()

"""
