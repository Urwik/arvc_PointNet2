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

from models import pointnet2_bin_seg
from arvc_Utils.Datasets import PLYDataset
from arvc_Utils.pointcloudUtils import np2ply

def test(device_, dataloader_, model_, loss_fn_, weights_):
    # TEST
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst = [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, filename_) in enumerate(tqdm(dataloader_)):
            data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.int64)
            pred_prob, abstract_points = model_(data.transpose(1, 2))
            pred_prob = pred_prob.flatten(start_dim=0, end_dim=1) # NLLLoss no se le pueden pasar batches
            pred_label = torch.argmax(pred_prob, dim=1).flatten()

            label = label.flatten()

            avg_loss = loss_fn_(pred_prob, label, weights_)
            loss_lst.append(avg_loss.item())

            avg_f1, avg_pre, avg_rec, conf_m = compute_metrics(label, pred_label)

            f1_lst.append(avg_f1)
            pre_lst.append(avg_pre)
            rec_lst.append(avg_rec)
            conf_m_lst.append(conf_m)

            if SAVE_PRED_CLOUDS:
                save_pred_as_ply(data, pred_label, PRED_CLOUDS_DIR, filename_)

            # current_clouds += data.size(0)
            #
            # if batch % 1 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
            #     print(f'  [Batch: {current_clouds}/{len(dataloader_.dataset)}],'
            #           f'  [File: {str(filename_)}],'
            #           f'  [F1 score: {avg_f1:.4f}],'
            #           f'  [Precision score: {avg_pre:.4f}],'
            #           f'  [Recall score: {avg_rec:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst


def compute_metrics(label, pred):

    pred = pred.cpu().numpy()
    label = label.cpu().numpy().astype(int)

    f1_score_ = metrics.f1_score(label, pred)
    precision_ = metrics.precision_score(label, pred)
    recall_ = metrics.recall_score(label, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(label, pred, labels=[0,1]).ravel()

    return f1_score_, precision_, recall_, (tn, fp, fn, tp)



def save_pred_as_ply(data_, pred_fix_, out_dir_, filename_):
    data_ = data_.detach().cpu().numpy()
    pred_fix_ = pred_fix_.detach().cpu().numpy()
    batch_size = np.size(data_, 0)
    n_points = np.size(data_, 1)

    feat_xyzlabel = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u4')]

    for i in range(batch_size):
        xyz = data_[i][:, [0,1,2]]
        actual_pred = pred_fix_[:,None]
        cloud = np.hstack((xyz, actual_pred))
        filename = filename_[0]
        np2ply(cloud, out_dir_, filename, features=feat_xyzlabel, binary=True)


if __name__ == '__main__':
    start_time = datetime.now()

    # --------------------------------------------------------------------------------------------#
    # GET CONFIGURATION PARAMETERS
    MODEL_DIR = 'bs_xyz_bce_vf_loss/'

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
                         transform = None)

    WEIGHTS = torch.Tensor(dataset.weights).to(DEVICE)
    # INSTANCE DATALOADER
    test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)


    # SELECT DEVICE TO WORK WITH
    device = torch.device(DEVICE)
    model = pointnet2_bin_seg.get_model(num_classes=OUTPUT_CLASSES,
                                             n_feat=len(FEATURES)).to(device)
    loss_fn = pointnet2_bin_seg.get_loss().to(device)


    # MAKE DIR WHERE TO SAVE THE CLOUDS
    PRED_CLOUDS_DIR = os.path.join(MODEL_PATH, "pred_clouds")
    if not os.path.exists(PRED_CLOUDS_DIR):
        os.makedirs(PRED_CLOUDS_DIR)

    # LOAD TRAINED MODEL
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best_model.pth'), map_location=device))
    # threshold = np.load(MODEL_PATH + f'/threshold.npy')
    # THRESHOLD = np.mean(threshold[-1])

    print('-'*50)
    print('TESTING ON: ', device)
    results = test(device_=device,
                   dataloader_=test_dataloader,
                   model_=model,
                   loss_fn_=loss_fn,
                   weights_=WEIGHTS)

    f1_score = np.mean(np.array(results[1]))
    precision = np.mean(np.array(results[2]))
    recall = np.mean(np.array(results[3]))
    confusion_matrix_list = np.array(results[4])
    conf_matrix = np.mean(confusion_matrix_list, axis=0)

    print('\n\n')
    # print(f'Threshold: {THRESHOLD}')
    print(f'Avg F1_score: {f1_score}')
    print(f'Avg Precision: {precision}')
    print(f'Avg Recall: {recall}')
    print(f'TN: {conf_matrix[0]}, FP: {conf_matrix[1]}, FN: {conf_matrix[2]}, TP: {conf_matrix[3]}')
    print("Done!")
