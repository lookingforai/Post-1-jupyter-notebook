# The following functions will transform a dataset of LABELED images
# into a .h5 or .hdf5 file.

from random import shuffle
import glob
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt


def img_get_paths_labels(img_path, key_words):
    """

    :param img_path: location the images that we will use to train (e.g. 'Cat vs Dog/train/*.jpg')
    :param key_words: string to allow us to label the data (e.g. 'cat' if name of cat photos contains cat)
    :return:
        paths: list of path names
        labels: list of labels of the path names

    """

    # Read addresses from the img_path folder
    # Return a possibly-empty list of path names that match img_path,
    # which must be a string containing a path specification.

    paths = glob.glob(img_path)

    # Vector of labels, depending on if key_word is contained in a path
    # 1 = key_word
    labels = []
    for path in paths:
        i = 0
        for key_word in key_words:
            if key_word in path:
                labels.append(i)
            i = i + 1

    return paths, labels


def shuffle_paths_labels(paths, labels, shuffle_data):
    """

    :param paths: list of path names
    :param labels: list of labels of the path names
    :param shuffle_data: True if we want to shuffle
    :return:
        paths: list of path names
        labels: list of labels of the path names
    """

    if shuffle_data:
        c = list(zip(paths, labels))
        shuffle(c)
        paths, labels = zip(*c)

    return paths, labels


def train_dev_test_paths(paths, labels, train_dev_test):
    """

    :param paths: list of path names
    :param labels: list of labels of path names
    :param train_dev_test: (3,1) vector of [train/total, dev/total, test/total]
    :return: dict_path_labels: a dictionary containing the lists with all the paths and labels
    """

    # Get percentages from the train_dev_test vector

    train_end = train_dev_test[0]
    dev_end = train_dev_test[0] + train_dev_test[1]

    # Divide the paths and labels

    train_paths = paths[0:int(train_end * len(paths))]
    train_labels = labels[0:int(train_end * len(labels))]
    dev_paths = paths[int(train_end * len(paths)):int(dev_end * len(paths))]
    dev_labels = labels[int(train_end * len(paths)):int(dev_end * len(paths))]
    test_paths = paths[int(dev_end * len(paths)):]
    test_labels = labels[int(dev_end * len(labels)):]

    dict_path_labels = {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "dev_paths": dev_paths,
        "dev_labels": dev_labels,
        "test_paths": test_paths,
        "test_labels": test_labels
    }

    return dict_path_labels


def create_h5_file(hdf5_path, h_pixels, w_pixels, dict_paths_labels):
    """

    :param hdf5_path:
    :param h_pixels:
    :param w_pixels:
    :param dict_paths_labels: a dictionary containing the paths and labels for train, dev, and test sets.
    :return:
    """

    # We first get the contents from the dictionary

    train_paths = dict_paths_labels["train_paths"]
    train_labels = dict_paths_labels["train_labels"]
    dev_paths = dict_paths_labels["dev_paths"]
    dev_labels = dict_paths_labels["dev_labels"]
    test_paths = dict_paths_labels["test_paths"]
    test_labels = dict_paths_labels["test_labels"]

    # Set the data shape compatible with TensorFlow

    train_shape = (len(train_paths), h_pixels, w_pixels, 3)
    dev_shape = (len(dev_paths), h_pixels, w_pixels, 3)
    test_shape = (len(test_paths), h_pixels, w_pixels, 3)

    # Open a hdf5 file and create arrays (if it does not work, add np.int8 after all the create_datase)

    with h5py.File(hdf5_path, mode='w') as hdf5_file:

        hdf5_file.create_dataset("train_set_x", train_shape, np.uint8)
        hdf5_file.create_dataset("train_set_y", (len(train_labels),), np.uint8)

        hdf5_file.create_dataset("dev_set_x", dev_shape, np.uint8)
        hdf5_file.create_dataset("dev_set_y", (len(dev_labels),), np.uint8)

        hdf5_file.create_dataset("test_set_x", test_shape, np.uint8)
        hdf5_file.create_dataset("test_set_y", (len(test_labels),), np.uint8)

        hdf5_file["train_set_y"][...] = train_labels
        hdf5_file["dev_set_y"][...] = dev_labels
        hdf5_file["test_set_y"][...] = test_labels

        # loop over train addresses

        for i in range(len(train_paths)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Train data: {}/{}'.format(i, len(train_paths)))
            # read an image and resize to (224, 224)
            # cv2 load images as BGR, convert it to RGB
            path = train_paths[i]
            img = cv2.imread(path)
            img = cv2.resize(img, (w_pixels, h_pixels), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here

            # save the image and calculate the mean so far
            hdf5_file["train_set_x"][i, :] = img

        # loop over validation addresses
        for i in range(len(dev_paths)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Development data: {}/{}'.format(i, len(dev_paths)))
            # read an image and resize to (224, 224)
            # cv2 load images as BGR, convert it to RGB
            path = dev_paths[i]
            img = cv2.imread(path)
            img = cv2.resize(img, (w_pixels, h_pixels), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here

            # save the image
            hdf5_file["dev_set_x"][i, :] = img
        # loop over test addresses
        for i in range(len(test_paths)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Test data: {}/{}'.format(i, len(test_paths)))
            # read an image and resize to (224, 224)
            # cv2 load images as BGR, convert it to RGB
            path = test_paths[i]
            img = cv2.imread(path)
            img = cv2.resize(img, (w_pixels, h_pixels), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # add any image pre-processing here

            # save the image
            hdf5_file["test_set_x"][i, :] = img


def create_hdf5_file_from_path(hdf5_path, img_path, key_words, shuffle_data, h_pixels, w_pixels, train_dev_test):
    """

    :param hdf5_path:
    :param img_path:
    :param key_words:
    :param shuffle_data:
    :param h_pixels:
    :param w_pixels:
    :param train_dev_test:
    :return:
    """

    # Get the paths and labels of the images
    paths, labels = img_get_paths_labels(img_path, key_words)

    # Shuffle the images
    paths, labels = shuffle_paths_labels(paths, labels, shuffle_data)

    # Get a dictionary of the divisions of train/dev/test
    dict_paths_labels = train_dev_test_paths(paths, labels, train_dev_test)

    # Create the hdf5 file
    create_h5_file(hdf5_path, h_pixels, w_pixels, dict_paths_labels)


def load_dataset(hdf5_path):
    """

    :param hdf5_path: path with the file stored
    :return: dataset_dictionary: dictionary that contains all the data
    """
    train_dataset = h5py.File(hdf5_path, "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    dev_set_x_orig = np.array(train_dataset["dev_set_x"][:])  # your train set features
    dev_set_y_orig = np.array(train_dataset["dev_set_y"][:])  # your train set labels
    dev_set_y_orig = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))

    test_set_x_orig = np.array(train_dataset["test_set_x"][:])  # your train set features
    test_set_y_orig = np.array(train_dataset["test_set_y"][:])  # your train set labels
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    dataset_dictionary = {
        "train_set_x_orig": train_set_x_orig,
        "train_set_y_orig": train_set_y_orig,
        "dev_set_x_orig": dev_set_x_orig,
        "dev_set_y_orig": dev_set_y_orig,
        "test_set_x_orig": test_set_x_orig,
        "test_set_y_orig": test_set_y_orig
    }

    return dataset_dictionary


def gallery(img_array, height):
    """
    Creates a grid of images of size height*height
    :param img_array: array that contains all the images
    :param height: height of the gallery
    :return: img_grid
    """
    img_grid = img_array[0]
    for j in range(1, height):
        img_grid = np.concatenate([img_grid, img_array[j]], axis=1)
    for i in range(1, height):
        img_row = img_array[i*height]
        for j in range(1, height):
            img_row = np.concatenate([img_row, img_array[i*height+j]], axis=1)
        img_grid = np.concatenate([img_grid, img_row], axis=0)
    return img_grid

