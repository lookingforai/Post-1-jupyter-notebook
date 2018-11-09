from data_preparation.data_utils import create_hdf5_file_from_path, load_dataset, gallery
import matplotlib.pyplot as plt
import numpy as np

hdf5_path = '../images/data_1/data_1.hdf5'
img_path = "../images/data_1/*/*"
key_words = ["atleti", "betis", "chelsea"]
shuffle_data = True
train_dev_test = [0.9, 0, 0.1]
h_pixels = 112
w_pixels = 112


create_hdf5_file_from_path(hdf5_path, img_path, key_words, shuffle_data, h_pixels, w_pixels, train_dev_test)
dataset_dictionary = load_dataset(hdf5_path)
X_train_orig = dataset_dictionary["train_set_x_orig"]
Y_train_orig = dataset_dictionary["train_set_y_orig"]

"""
index = 4
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
plt.show() 
"""

plt.imshow(gallery(X_train_orig, 10))
plt.show()




