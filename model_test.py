import argparse

import numpy as np
import pandas as pd
from sklearn import metrics

from dataset import DataSet
from feature_transform import get_vector, distance, mean_vectors
from model import load_model


def model_simple_test(model_path, file_path, output_shape, sample_rate, process_class, model_type):
    num_of_voice_to_be_anchor = 3
    model = load_model(model_path, model_type)
    dataset = DataSet(file_dir=file_path, output_shape=output_shape, sample_rate=sample_rate)
    x, y = dataset.get_train_data(process_class=process_class)

    anchor = []
    done = 0
    index = 0
    while done < num_of_voice_to_be_anchor:
        if y[index] == 0:
            anchor.append(x.pop(index))
            y.pop(index)
            done += 1
        index += 1
    anchor_vectors = get_vector(np.array(anchor), model)
    anchor_vector = mean_vectors(anchor_vectors)

    sample_vectors = get_vector(np.array(x), model)
    result = distance(anchor_vector, sample_vectors, dis_type=0)

    # k = pd.Series(result)
    # k.mean()
    # k.hist()
    # plt.show()

    acc_score(y, result)
    fpr, tpr, thresholds = metrics.roc_curve(y, result, pos_label=1)
    # print(fpr, tpr, thresholds)
    # pd.DataFrame(data=np.array([fpr, tpr, thresholds]).T, columns=["fpr", "tpr", "thr"])
    auc = metrics.auc(fpr, tpr)
    print("auc:", auc)
    return auc


def leave_device_auc_test(model_path, file_path, output_shape, sample_rate, process_class, model_type):
    num_of_voice_to_be_anchor = 3
    model = load_model(model_path, model_type)
    dataset = DataSet(file_dir=file_path, output_shape=output_shape, sample_rate=sample_rate)
    x, y = dataset.get_train_data(process_class=process_class)
    predict_class = []

    class_num = len(set(y))
    chosen_class = np.random.choice(range(class_num), int(class_num / 2))

    anchor_voice = dict(zip(chosen_class, [[] for _ in range(len(chosen_class))]))
    for cls in chosen_class:
        num = 0
        for i in range(len(y)):
            if y[i] == cls:
                anchor_voice[cls].append(x[i])
                num += 1
                if num == num_of_voice_to_be_anchor:
                    break

    anchor_voice = [mean_vectors(model.predict(np.array(anchor_voice[cls]))) for cls in anchor_voice.keys()]

    # reset class
    for i in range(len(y)):
        if y[i] in chosen_class:
            y[i] = 1
        else:
            y[i] = 0

    x = model.predict(np.array(x))


    # calculate distance
    dis = []
    for i in range(x.shape[0]):
        dis.append(min(distance(x[i], np.array(anchor_voice))))

    acc_score(y, dis)
    fpr, tpr, thresholds = metrics.roc_curve(y, dis, pos_label=1)
    # print(fpr, tpr, thresholds)
    # pd.DataFrame(data=np.array([fpr, tpr, thresholds]).T, columns=["fpr", "tpr", "thr"])
    auc = metrics.auc(fpr, tpr)
    print("auc:", auc)

    # # calculate distance and class
    # dis = []
    # for i in range(x.shape[0]):
    #     full_distance = distance(x[i], np.array(anchor_voice))
    #     min_dis = min(full_distance)
    #     cls = full_distance.index(min_dis)
    #     dis.append(min_dis)
    #     predict_class.append(cls)
    #
    # # create dataframe
    # index = list(range(len(x)))
    # df = pd.DataFrame(data={"ind": index, "dis": dis, "cls": predict_class})
    # pre = get_real_prediction(df)
    # pre_list = list(zip(pre.keys(), pre.values()))
    # pre_list = sorted(pre_list, key=lambda x:[0])
    # pre = [x[1] for x in pre_list]
    #
    # print(metrics.accuracy_score(y, pre))
    # acc_score(y, dis)
    # fpr, tpr, thresholds = metrics.roc_curve(y, pre, pos_label=1)
    # # print(fpr, tpr, thresholds)
    # # pd.DataFrame(data=np.array([fpr, tpr, thresholds]).T, columns=["fpr", "tpr", "thr"])
    # auc = metrics.auc(fpr, tpr)
    # print("auc:", auc)



def get_real_prediction(cls_pd):
    final = {}
    for cls, df in cls_pd.groupby("cls"):
        print(cls, df.shape)
        median = np.median(df["dis"].values)
        for index, row in df.iterrows():
            if row.dis <= median:
                final[row.ind] = 1
            else:
                final[row.ind] = 0
    return final


def acc_score(y, y_prediction):
    s_l = sorted(list(zip(y, y_prediction)), key=lambda x: x[1], reverse=True)
    true_pos = 0
    true_neg = 0
    for i in range(len(s_l)):
        if i < len(s_l) / 2 and s_l[i][0] == 1:
            true_pos += 1
        elif i >= len(s_l) / 2 and s_l[i][0] == 0:
            true_neg += 1

    print("acc:", (true_neg + true_pos) / len(s_l))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("speaker recognition", fromfile_prefix_chars='@')
    parser.add_argument('--file_dir', type=str, help='Directory to load data.')
    parser.add_argument('--model_path', type=str, help='Directory to save model.')
    parser.add_argument('-s', '--output_shape', type=int, nargs=2, default=[32, 1024], help='shape')
    parser.add_argument('--hidden_size', type=int, default=64, help='The hidden full connected layer size')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=10, help='Number of epoch to train')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Number of training examples processed per step')
    # parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sample rate of wave')
    parser.add_argument('-c', '--class_num', type=int, default=103, help='class num of voice')
    parser.add_argument('-pc', '--process_class', type=int, default=0, help='class of process\' way')
    parser.add_argument('-mt', '--model_type', type=int, default=0,
                        help='type of model.0:res_plus_transformer; 1.simple_cnn; 2.res_net')
    parser.add_argument('-n', '--net_depth', type=int, default=1,
                        help='net depth of res_net')

    args = parser.parse_args()

    model_path = args.model_path
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    epochs = args.epochs_to_train
    file_dir = args.file_dir
    output_shape = args.output_shape
    class_num = args.class_num
    process_class = args.process_class
    model_type = args.model_type
    net_depth = args.net_depth

    leave_device_auc_test(model_path=model_path, file_path=file_dir, output_shape=output_shape, sample_rate=sample_rate,
                          process_class=process_class, model_type=model_type)
