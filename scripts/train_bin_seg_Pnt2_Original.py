import numpy as np
import os
import shutil
import math
from torch.utils.data import DataLoader
import torch
import socket
import sys
import sklearn.metrics as metrics
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# IMPORTS PATH TO THE PROJECT
current_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pycharm_projects_path = os.path.dirname(current_project_path)
# IMPORTS PATH TO OTHER PYCHARM PROJECTS
sys.path.append(current_project_path)
sys.path.append(pycharm_projects_path)

from arvc_Utils.Datasets import PLYDataset
from models import pointnet2_bin_seg


def train(device_, train_loader_, model_, loss_fn_, optimizer_, weights_):
    loss_lst = []
    current_clouds = 0

    # TRAINING
    print('-' * 50)
    print('TRAINING')
    print('-'*50)
    model_.train()
    for batch, (data, label, _) in enumerate(train_loader_):
        data, label = data.to(device_, dtype=torch.float32), label.to(device_, dtype=torch.int64)
        pred_prob, abstract_points = model_(data.transpose(1, 2))
        pred_prob = pred_prob.flatten(start_dim=0, end_dim=1)

        label = label.flatten()

        avg_train_loss_ = loss_fn_(pred_prob, label, weights_)
        loss_lst.append(avg_train_loss_.item())

        optimizer_.zero_grad()
        avg_train_loss_.backward()
        optimizer_.step()

        current_clouds += data.size(0)

        if batch % 1 == 0 or data.size(0) < train_loader_.batch_size:  # print every (% X) batches
            print(f' - [Batch: {current_clouds}/{len(train_loader_.dataset)}],'
                  f' / Train Loss: {avg_train_loss_:.4f}')

    return loss_lst


def valid(device_, dataloader_, model_, loss_fn_, weights_):

    # VALIDATION
    print('-' * 50)
    print('VALIDATION')
    print('-'*50)
    model_.eval()
    f1_lst, pre_lst, rec_lst, loss_lst, conf_m_lst = [], [], [], [], []
    current_clouds = 0

    with torch.no_grad():
        for batch, (data, label, _) in enumerate(dataloader_):
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

            current_clouds += data.size(0)

            if batch % 1 == 0 or data.size()[0] < dataloader_.batch_size:  # print every 10 batches
                print(f'  [Batch: {current_clouds}/{len(dataloader_.dataset)}]'
                      f'  [Loss: {avg_loss:.4f}]'
                      f'  [Precision: {avg_pre:.4f}]'
                      f'  [Recall: {avg_rec:.4f}'
                      f'  [F1 score: {avg_f1:.4f}]')

    return loss_lst, f1_lst, pre_lst, rec_lst, conf_m_lst


def compute_metrics(label, pred):

    pred = pred.cpu().numpy()
    label = label.cpu().numpy().astype(int)

    f1_score = metrics.f1_score(label, pred)
    precision_ = metrics.precision_score(label, pred)
    recall_ = metrics.recall_score(label, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(label, pred, labels=[0,1]).ravel()

    return f1_score, precision_, recall_, (tn, fp, fn, tp)


def compute_best_threshold(pred_, gt_):
    trshld_per_cloud = []
    method_ = THRESHOLD_METHOD
    for cloud in range(len(pred_)):
        if method_ == "roc":
            fpr, tpr, thresholds = metrics.roc_curve(gt_[cloud], pred_[cloud])
            gmeans = np.sqrt(tpr * (1 - fpr))
            index = np.argmax(gmeans)
            trshld_per_cloud.append(thresholds[index])

        elif method_ == "pr":
            precision_, recall_, thresholds = metrics.precision_recall_curve(gt_[cloud], pred_[cloud])
            f1_score_ = (2 * precision_ * recall_) / (precision_ + recall_)
            index = np.argmax(f1_score_)
            trshld_per_cloud.append(thresholds[index])

        elif method_ == "tuning":
            thresholds = np.arange(0.0, 1.0, 0.0001)
            f1_score_ = np.zeros(shape=(len(thresholds)))
            for index, elem in enumerate(thresholds):
                prediction_ = np.where(pred_[cloud] > elem, 1, 0).astype(int)
                f1_score_[index] = metrics.f1_score(gt_[cloud], prediction_)

            index = np.argmax(f1_score_)
            trshld_per_cloud.append(thresholds[index])
        else:
            print('Error in the name of the method to use for compute best threshold')

    return sum(trshld_per_cloud)/len(trshld_per_cloud)


if __name__ == '__main__':

    Files = ['xyz_NLLLoss_pr.yaml']

    for configFile in Files:
        # HYPERPARAMETERS
        start_time = datetime.now()

        # --------------------------------------------------------------------------------------------#
        # GET CONFIGURATION PARAMETERS
        CONFIG_FILE = configFile
        config_file_abs_path = os.path.join(current_project_path, 'config', CONFIG_FILE)
        with open(config_file_abs_path) as file:
            config = yaml.safe_load(file)

        # DATASET
        TRAIN_DIR = config["TRAIN_DIR"]
        VALID_DIR = config["VALID_DIR"]
        USE_VALID_DATA = config["USE_VALID_DATA"]
        OUTPUT_DIR = config["OUTPUT_DIR"]
        TRAIN_SPLIT = config["TRAIN_SPLIT"]
        FEATURES = config["FEATURES"]
        LABELS = config["LABELS"]
        NORMALIZE = config["NORMALIZE"]
        BINARY = config["BINARY"]
        # THRESHOLD_METHOS POSIBILITIES = cuda:X, cpu
        DEVICE = config["DEVICE"]
        BATCH_SIZE = config["BATCH_SIZE"]
        EPOCHS = config["EPOCHS"]
        LR = config["LR"]
        # MODEL
        OUTPUT_CLASSES = config["OUTPUT_CLASSES"]
        # THRESHOLD_METHOS POSIBILITIES = roc, pr, tuning
        THRESHOLD_METHOD = config["THRESHOLD_METHOD"]
        # TERMINATION_CRITERIA POSIBILITIES = loss, precision, f1_score
        TERMINATION_CRITERIA = config["TERMINATION_CRITERIA"]
        EPOCH_TIMEOUT = config["EPOCH_TIMEOUT"]

        # --------------------------------------------------------------------------------------------#
        # CHANGE PATH DEPENDING ON MACHINE
        machine_name = socket.gethostname()
        if machine_name == 'arvc-Desktop':
            TRAIN_DATA = os.path.join('/media/arvc/data/datasets', TRAIN_DIR)
            VALID_DATA = os.path.join('/media/arvc/data/datasets', VALID_DIR)
        else:
            TRAIN_DATA = os.path.join('/home/arvc/Fran/data/datasets', TRAIN_DIR)
            VALID_DATA = os.path.join('/home/arvc/Fran/data/datasets', VALID_DIR)
        # --------------------------------------------------------------------------------------------#
        # CREATE A FOLDER TO SAVE TRAINING
        OUT_DIR = os.path.join(current_project_path, OUTPUT_DIR)
        input_features = ""
        if FEATURES == [0,1,2]:
            input_features = "xyz"
        elif FEATURES == [0,1,2,7]:
            input_features = "xyzc"
        elif FEATURES == [0,1,2,4,5,6]:
            input_features = "xyzn"
        else:
            input_features = "???"

        folder_name = "bs" + '_' + input_features + '_' + datetime.today().strftime('%y%m%d%H%M')
        OUT_DIR = os.path.join(OUT_DIR, folder_name)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        shutil.copyfile(config_file_abs_path, os.path.join(OUT_DIR, 'config.yaml'))

        # ------------------------------------------------------------------------------------------------------------ #
        # INSTANCE DATASET
        train_dataset = PLYDataset(root_dir=TRAIN_DATA,
                                   features=FEATURES,
                                   labels=LABELS,
                                   normalize=NORMALIZE,
                                   binary=BINARY,
                                   transform=None)

        WEIGHTS = torch.Tensor(train_dataset.weights).to(DEVICE)

        if USE_VALID_DATA:
            valid_dataset = PLYDataset(root_dir=VALID_DATA,
                                       features=FEATURES,
                                       labels=LABELS,
                                       normalize=NORMALIZE,
                                       binary=BINARY,
                                       transform=None)
        else:
            # SPLIT VALIDATION AND TRAIN
            train_size = math.floor(len(train_dataset) * TRAIN_SPLIT)
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(74))

        # INSTANCE DATALOADERS
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=10,
                                      shuffle=True, pin_memory=True, drop_last=True)

        # ------------------------------------------------------------------------------------------------------------ #
        # SELECT MODEL
        device = torch.device(DEVICE)
        model = pointnet2_bin_seg.get_model(num_classes=OUTPUT_CLASSES,
                                                 n_feat=len(FEATURES)).to(device)
        loss_fn = pointnet2_bin_seg.get_loss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # ------------------------------------------------------------------------------------------------------------ #
        # --- TRAIN LOOP --------------------------------------------------------------------------------------------- #
        print('TRAINING ON: ', device)
        epoch_timeout_count = 0

        if TERMINATION_CRITERIA == "loss":
            best_val = 1
        else:
            best_val = 0

        f1, precision, recall, conf_matrix, train_loss, valid_loss, threshold = [], [], [], [], [], [], []

        for epoch in range(EPOCHS):
            print(f"EPOCH: {epoch} {'-' * 50}")
            epoch_start_time = datetime.now()

            train_results = train(device_=device,
                                  train_loader_=train_dataloader,
                                  model_=model,
                                  loss_fn_=loss_fn,
                                  optimizer_=optimizer,
                                  weights_=WEIGHTS)

            valid_results = valid(device_=device,
                                  dataloader_=valid_dataloader,
                                  model_=model,
                                  loss_fn_=loss_fn,
                                  weights_=WEIGHTS)

            # GET RESULTS
            train_loss.append(train_results)
            valid_loss.append(valid_results[0])
            f1.append(valid_results[1])
            precision.append(valid_results[2])
            recall.append(valid_results[3])
            conf_matrix.append(valid_results[4])

            print('-' * 50)
            print('DURATION:')
            print('-' * 50)
            epoch_end_time = datetime.now()
            print('Epoch Duration: {}'.format(epoch_end_time-epoch_start_time))

            # SAVE MODEL AND TEMINATION CRITERIA
            if TERMINATION_CRITERIA == "loss":
                last_val = np.mean(valid_results[0])
                if last_val < best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < EPOCH_TIMEOUT:
                    epoch_timeout_count += 1
                else:
                    break
            elif TERMINATION_CRITERIA == "precision":
                last_val = np.mean(valid_results[2])
                if last_val > best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < EPOCH_TIMEOUT:
                    epoch_timeout_count += 1
                else:
                    break
            elif TERMINATION_CRITERIA == "f1_score":
                last_val = np.mean(valid_results[1])
                if last_val > best_val:
                    torch.save(model.state_dict(), OUT_DIR + f'/best_model.pth')
                    best_val = last_val
                    epoch_timeout_count = 0
                elif epoch_timeout_count < EPOCH_TIMEOUT:
                    epoch_timeout_count += 1
                else:
                    break
            else:
                print("WRONG TERMINATION CRITERIA")
                exit()


        # SAVE RESULTS
        np.save(OUT_DIR + f'/train_loss', np.array(train_loss))
        np.save(OUT_DIR + f'/valid_loss', np.array(valid_loss))
        np.save(OUT_DIR + f'/f1_score', np.array(f1))
        np.save(OUT_DIR + f'/precision', np.array(precision))
        np.save(OUT_DIR + f'/recall', np.array(recall))
        np.save(OUT_DIR + f'/conf_matrix', np.array(conf_matrix))

        end_time = datetime.now()
        print('Total Training Duration: {}'.format(end_time-start_time))
        print("Training Done!")
