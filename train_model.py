import threading
import time
import numpy as np
from build_model import falling_detection_model
from keras.callbacks import TensorBoard, ModelCheckpoint


def train_on_real_dataset_proposed_model():
    """
    spot_1 real 10 frames data: 1189 // train 700
    spot_2 real 10 frames data: 326
    spot_3 real 10 frames data: 649
    spot_4 real 10 frames data: 694

    """
    # tain on spot1 350 + spot4 350

    batch = 4
    gen = realdata_generator(batch_sz=batch)

    tenboard = TensorBoard(log_dir='log/C2C_AE_model_10_3_spot184', write_images=1)

    checkpoint = ModelCheckpoint('weights/C2C_AE_model_10_3_10frame_by_spot124_real_data_set.h5',
                                 monitor='loss', verbose=1, save_best_only=True)

    valid_data_1 = np.load('dataset_feature/spot_1_test_real_falling.npy')
    valid_data_2 = np.load('dataset_feature/spot_2_test_real_falling.npy')
    valid_data_3 = np.load('dataset_feature/spot_4_test_real_falling.npy')

    valid_data = np.concatenate((valid_data_1, valid_data_2, valid_data_3))
    model = falling_detection_model()  # my model

    model.summary()

    model.fit_generator(generator=gen, steps_per_epoch=700 // batch, epochs=100,
                        validation_data=(valid_data/255., valid_data/255.),
                        max_queue_size=50, workers=4, callbacks=[tenboard, checkpoint])


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def onlinedata_generator(batch_sz=2):
    start = 0
    data_folders = ['dataset_feature/DataSet/ped2_20frame_gray_normal_uint8.npy',
                    'dataset_feature/DataSet/avenue_20frame_gray_normal_10-16_uint8.npy',
                    'dataset_feature/DataSet/ped1_20frame_gray_normal_uint8.npy',
                    'dataset_feature/DataSet/avenue_20frame_gray_normal_1-9_uint8.npy']

    while True:
        for data_root in data_folders:
            data = np.load(data_root)
            num_data = data.shape[0]
            # print('\n', data_root, num_data)

            while True:
                end = start + batch_sz
                # batch_data = np.squeeze(data[start:end], axis=-1)
                batch_data = data[start:end, :10]/255.

                start = end
                yield (batch_data, batch_data)

                if start >= num_data:
                    start = 0
                    break


@threadsafe_generator
def realdata_generator(batch_sz=2):
    """
    spot_1 real 10 frames data: 1189 // train 700
    spot_2 real 10 frames data: 326
    spot_3 real 10 frames data: 649
    spot_4 real 10 frames data: 694

    """
    start = 0
    data_root_1 = 'dataset_feature/spot_1_real_normal.npy'
    data_root_2 = 'dataset_feature/spot_2_real_normal.npy'
    data_root_3 = 'dataset_feature/spot_4_real_normal.npy'

    print('train on ' + '124')
    data_1 = np.load(data_root_1)
    data_2 = np.load(data_root_2)
    data_3 = np.load(data_root_3)

    np.random.shuffle(data_1)
    np.random.shuffle(data_2)
    np.random.shuffle(data_3)

    data = np.concatenate((data_1[:233], data_2[:233], data_3[:234]))

    np.random.shuffle(data)

    # data = data[:700]
    num_data = data.shape[0]
    print('num train data: ', num_data)

    while True:
        end = start + batch_sz
        batch_data = data[start:end]/255.
        start = end
        yield (batch_data, batch_data)
        if start >= num_data:
            start = 0


if __name__ == '__main__':
    train_on_real_dataset_proposed_model()
