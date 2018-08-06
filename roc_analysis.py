from sklearn import metrics
from sklearn.metrics import auc
from build_model import falling_detection_model
import numpy as np
import matplotlib.pyplot as plt


def regularity_score(reconstructed_error):

    num_frame = len(reconstructed_error)

    max_error = np.max(reconstructed_error)
    min_error = np.min(reconstructed_error)
    print('max_error', max_error)
    print('min_error', min_error)

    r_score = np.zeros(num_frame)
    for i in range(num_frame):
        r_score[i] = (reconstructed_error[i] - min_error)/max_error

    return r_score


def analysis(model, model_name, normal_data, falling_data, color='r'):
    model.summary()
    print('normal_data: ',normal_data.shape)
    print('falling_data: ',falling_data.shape)


    fall_memory = []
    normal_memory = []
    all_reconstruct_error = []
    y_label = np.zeros(falling_data.shape[0] + normal_data.shape[0])
    y_label[:falling_data.shape[0]] = 1

    pre = np.zeros_like(falling_data)
    # use for loop because of OOM problem

    for i in range(falling_data.shape[0]):
        pre[i] = model.predict(np.expand_dims(falling_data[i], axis=0))

    for i in range(len(pre)):
        reconstruct_error = np.mean(np.power(falling_data[i] - pre[i], 2))
        all_reconstruct_error.append(reconstruct_error)
        fall_memory.append(reconstruct_error)

    pre = np.zeros_like(normal_data)
    # use for loop because of OOM problem

    for i in range(normal_data.shape[0]):
        pre[i] = model.predict(np.expand_dims(normal_data[i], axis=0))

    for i in range(len(pre)):
        reconstruct_error = np.mean(np.power(normal_data[i] - pre[i], 2))
        all_reconstruct_error.append(reconstruct_error)
        normal_memory.append(reconstruct_error)

    r_score = regularity_score(all_reconstruct_error)

    fpr, tpr, thresholds = metrics.roc_curve(y_label, all_reconstruct_error)

    print('tpr: ', tpr)
    print('fpr: ', fpr)


    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    print('Threshold: ', eer_threshold)

    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color, label=model_name + ' AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

np.random.seed(666)

if __name__ == '__main__':

    spot = 3

    plt.figure()

    normal_data = np.load('dataSet/test/spot_%d_test_real_normal.npy' % spot)
    abnormal_data = np.load('dataSet/test/spot_%d_test_real_falling.npy' % spot)

    model = falling_detection_model()

    model.load_weights('weights/model_train_on_spot1.h5')
    analysis(model, 'camera_A', normal_data/255., abnormal_data/255., color='r')

    plt.show()