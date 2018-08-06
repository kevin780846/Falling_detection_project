import keras
import scipy
import time
import os
from build_model import falling_detection_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz


def rgb2gray(rgb, size):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return scipy.misc.imresize(gray.astype(np.uint8), size=size)


def full_image_flow(model_name, model, falling_folder_path, thresh, image_size=(240, 320), squeeze=True, color='r'):
    def slinding_window(a, window=10, step=1, axis=0):
        # print a.shape, a.strides
        shape = a.shape[:axis] + ((a.shape[axis] - window) // step + 1, window) + a.shape[axis + 1:]
        strides = a.strides[:axis] + (a.strides[axis] * step, a.strides[axis]) + a.strides[axis + 1:]
        # print shape, strides
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


    # falling ....

    total_frame = len(os.listdir(falling_folder_path))

    all_image = np.zeros((total_frame, *image_size, 1))

    imgs = sorted(os.listdir(falling_folder_path))[:total_frame]

    for index, img in enumerate(imgs):
        all_image[index] = np.expand_dims(rgb2gray(plt.imread(falling_folder_path + '/' + img), size=image_size),
                                          axis=-1)

    stride_img = slinding_window(all_image, step=1)

    if squeeze:
        stride_img = np.squeeze(stride_img, axis=-1)

    loss = []
    for i in range(stride_img.shape[0]):
        orig = stride_img[i] / 255.
        pre = model.predict(np.expand_dims(stride_img[i] / 255., axis=0))
        reconstruct_error = np.mean(np.power(orig - pre, 2))
        loss.append(reconstruct_error)

    plt.plot(range(len(loss)), loss, color, label=model_name)
    plt.hlines(thresh, 0,len(loss), 'r', label='th')

    plt.xlabel('Starting Frame')
    plt.ylabel('Reconstruct Error')
    plt.legend(loc='upper right')
    plt.title('Falling Detection Reconstruct Error Flow')

    # plt.plot(1-r_score, color, label=model_name)

    plt.ylim([0.005, 0.009])
    # plt.plot(loss, color, label=model_name)
    plt.legend(loc='upper right')
    plt.xlabel('Starting Frame')
    plt.ylabel('Reconstruct Error')
    plt.title('Falling Detection Reconstruct Error Flow')


if __name__ == '__main__':
    spot = 1
    video_num = 8

    falling_folder_path = 'experiment_video/Spot_%d/full_falling_%d' % (
    spot, video_num)
    normal_folder_path = 'experiment_video/Spot_%d/normal_2' % spot

    model = falling_detection_model()
    model.load_weights('weights/model_train_on_spot1.h5')

    # threshold value from roc_analysis.py
    # thresh index: [spot1, spot2, spot3, spot4]
    scen1_thresh = [0.0033070739255784735, 0.0067860934233935155, 0.0075616368197965545, 0.007180703010778931]
    scen3_thresh = [0.004799677139467654, 0.0046782219794217015, 0.007708327183386716, 0.005254429025773579]

    for index, spot in enumerate([3]):

        threshold = scen1_thresh[index+2]
        plt.figure()
        video_num = 11
        falling_folder_path = 'experiment_video/Spot_%d/full_falling_%d' % (spot, video_num)
        full_image_flow('R_E', model, falling_folder_path, threshold,
                        squeeze=False, image_size=(240, 320), color='b')

    model.load_weights('weights/model_train_on_spot124.h5')
    for index, spot in enumerate([3]):

        threshold = scen3_thresh[index+2]
        plt.figure()

        video_num = 11
        falling_folder_path = 'experiment_video/Spot_%d/full_falling_%d' % (spot, video_num)
        full_image_flow('R_E', model, falling_folder_path, threshold,
                        squeeze=False, image_size=(240, 320), color='b')

    plt.show()