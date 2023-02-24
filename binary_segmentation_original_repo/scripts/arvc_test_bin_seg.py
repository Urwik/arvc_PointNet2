import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import sklearn.metrics as metrics
import sys
import socket
import yaml
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# IMPORTS PATH TO THE PROJECT
current_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(current_project_path)
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_project_path)
sys.path.append(pycharm_projects_path)

from models import arvc_pointnet2_bin_seg
from arvc_Utils.Datasets import PLYDataset
from arvc_Utils.pointcloudUtils import np2ply

def test(device_, dataloader_, model_, loss_fn_):
    # TEST
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst, files_lst = [], [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, filename_) in enumerate(tqdm(dataloader_)):
            data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.float32)
            pred, abstract = model_(data.transpose(1, 2))
            m = torch.nn.Sigmoid()
            pred = m(pred)
            pred = torch.squeeze(pred, 0)

            avg_loss = loss_fn_(pred, label)
            loss_lst.append(avg_loss.item())
            files_lst.append(filename_)

            pred_fix, avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, pred)
            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            if SAVE_PRED_CLOUDS:
                save_pred_as_ply(data, pred_fix, PRED_CLOUDS_DIR, filename_)

            # current_clouds += data.size(0)
            #
            # if batch % 1 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
            #     print(f'  [Batch: {current_clouds}/{len(dataloader_.dataset)}],'
            #           f'  [File: {str(filename_)}],'
            #           f'  [F1 score: {avg_f1:.4f}],'
            #           f'  [Precision score: {avg_pre:.4f}],'
            #           f'  [Recall score: {avg_rec:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst, files_lst


def compute_metrics(label_, pred_):

    pred = pred_.cpu().numpy()
    label = label_.cpu().numpy().astype(int)
    trshld = THRESHOLD
    pred = np.where(pred > trshld, 1, 0).astype(int)

    f1_score_list = []
    precision_list = []
    recall_list =  []
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    batch_size = np.size(pred, 0)
    for i in range(batch_size):
        tmp_labl = label[i]
        tmp_pred = pred[i]

        f1_score = metrics.f1_score(tmp_labl, tmp_pred, average='binary')
        precision = metrics.precision_score(tmp_labl, tmp_pred, average='binary')
        recall = metrics.recall_score(tmp_labl, tmp_pred, average='binary')
        tn, fp, fn, tp = metrics.confusion_matrix(tmp_labl, tmp_pred, labels=[0,1]).ravel()

        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)

        f1_score_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)

    avg_f1_score = np.mean(np.array(f1_score_list))
    avg_precision = np.mean(np.array(precision_list))
    avg_recall = np.mean(np.array(recall_list))
    avg_tn = np.mean(np.array(tn_list))
    avg_fp = np.mean(np.array(fp_list))
    avg_fn = np.mean(np.array(fn_list))
    avg_tp = np.mean(np.array(tp_list))

    return pred, avg_f1_score, avg_precision, avg_recall, (avg_tn, avg_fp, avg_fn, avg_tp)


def save_pred_as_ply(data_, pred_fix_, out_dir_, filename_):
    data_ = data_.detach().cpu().numpy()
    batch_size = np.size(data_, 0)
    n_points = np.size(data_, 1)

    feat_xyzlabel = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u4')]

    for i in range(batch_size):
        xyz = data_[i][:, [0,1,2]]
        actual_pred = pred_fix_[i].reshape(n_points, 1)
        cloud = np.hstack((xyz, actual_pred))
        filename = filename_[0]
        np2ply(cloud, out_dir_, filename, features=feat_xyzlabel, binary=True)


def get_representative_clouds(f1_score_, precision_, recall_, files_list_):

    print('-'*50)
    print("Representative Clouds")

    max_f1 = np.max(f1_score_)
    min_f1 = np.min(f1_score_)
    max_pre = np.max(precision_)
    min_pre = np.min(precision_)
    max_rec = np.max(recall_)
    min_rec = np.min(recall_)

    max_f1_idx = list(f1_score_).index(max_f1.item())
    min_f1_idx = list(f1_score_).index(min_f1.item())
    max_pre_idx = list(precision_).index(max_pre.item())
    min_pre_idx = list(precision_).index(min_pre.item())
    max_rec_idx = list(recall_).index(max_rec.item())
    min_rec_idx = list(recall_).index(min_rec.item())

    print(f'Max f1 cloud: {files_list_[max_f1_idx]}')
    print(f'Min f1 cloud: {files_list_[min_f1_idx]}')
    print(f'Max precision cloud: {files_list_[max_pre_idx]}')
    print(f'Min precision cloud: {files_list_[min_pre_idx]}')
    print(f'Max recall cloud: {files_list_[max_rec_idx]}')
    print(f'Min recall cloud: {files_list_[min_rec_idx]}')


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # GET CONFIGURATION PARAMETERS
    # CONFIG_FILE = 'test_configuration.yaml'
    # MODEL_DIR = '230203_1433'
    MODEL_DIR = '230204_0120'

    config_file_abs_path = os.path.join(current_project_path, 'model_save', MODEL_DIR, 'config.yaml')
    with open(config_file_abs_path) as file:
        config = yaml.safe_load(file)

    # DATASET
    TEST_DIR= config["test"]["TEST_DIR"]
    FEATURES= config["FEATURES"]
    LABELS= config["LABELS"]
    NORMALIZE= config["NORMALIZE"]
    BINARY= config["BINARY"]
    # THRESHOLD_METHOS POSIBILITIES = cuda:X, cpu
    DEVICE= config["test"]["DEVICE"]
    BATCH_SIZE= config["test"]["BATCH_SIZE"]
    # MODEL
    MODEL_PATH= os.path.join(current_project_path, 'model_save', MODEL_DIR)
    # THRESHOLD= config["THRESHOLD"]
    OUTPUT_CLASSES= config["OUTPUT_CLASSES"]
    # LOSS = BCELoss()
    # RESULTS
    SAVE_PRED_CLOUDS= config["test"]["SAVE_PRED_CLOUDS"]
    # PRED_CLOUDS_DIR= config["test"]["PRED_CLOUDS_DIR"]

    # --------------------------------------------------------------------------------------------#
    # CHANGE PATH DEPENDING ON MACHINE
    machine_name = socket.gethostname()
    if machine_name == 'arvc-Desktop':
        TEST_DATA = os.path.join('/media/arvc/data/datasets', TEST_DIR)
    else:
        TEST_DATA = os.path.join('/home/arvc/Fran/data/datasets', TEST_DIR)
    # --------------------------------------------------------------------------------------------#
    # INSTANCE DATASET
    dataset = PLYDataset(root_dir = TEST_DATA,
                         features= FEATURES,
                         labels = LABELS,
                         normalize = NORMALIZE,
                         binary = BINARY,
                         compute_weights=False)

    # INSTANCE DATALOADER
    test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)

    # SELECT DEVICE TO WORK WITH
    device = torch.device(DEVICE)
    model = arvc_pointnet2_bin_seg.get_model(num_classes=OUTPUT_CLASSES,
                                             n_feat=len(FEATURES)).to(device)
    loss_fn = torch.nn.BCELoss()


    # MAKE DIR WHERE TO SAVE THE CLOUDS
    PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "pred_clouds")
    if not os.path.exists(PRED_CLOUDS_DIR):
        os.makedirs(PRED_CLOUDS_DIR)

    # LOAD TRAINED MODEL
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best_model.pth'), map_location=device))
    threshold = np.load(MODEL_PATH + f'/threshold.npy')
    THRESHOLD = np.mean(threshold[-1])

    print('-'*50)
    print('TESTING ON: ', device)
    results = test(device_=device,
                   dataloader_=test_dataloader,
                   model_=model,
                   loss_fn_=loss_fn)

    f1_score = np.array(results[1])
    precision = np.array(results[2])
    recall = np.array(results[3])
    confusion_matrix_list = np.array(results[4])
    conf_matrix = np.mean(confusion_matrix_list, axis=0)
    files_list = results[5]

    get_representative_clouds(f1_score, precision, recall, files_list)


    print('\n\n')
    print(f'Threshold: {THRESHOLD}')
    print(f'Avg F1_score: {np.mean(f1_score)}')
    print(f'Avg Precision: {np.mean(precision)}')
    print(f'Avg Recall: {np.mean(recall)}')
    print(f'TN: {conf_matrix[0]}, FP: {conf_matrix[1]}, FN: {conf_matrix[2]}, TP: {conf_matrix[3]}')
    print("Done!")
