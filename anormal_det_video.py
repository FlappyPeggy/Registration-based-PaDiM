import glob
import os
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from regpadim.resnet import resnet18
from regpadim.e2eutils import *
from regpadim.unet import *
import cv2
import datetime

MEAN = np.array([0.485, 0.456, 0.406])[:, None, None]
STD = np.array([0.229, 0.224, 0.225])[:, None, None]
IDX = np.array([47, 418, 70, 100, 204, 223, 131, 435, 24, 7, 444, 78, 224, 69, 314, 9, 103, 276, 81,
                109, 17, 263, 158, 397, 188, 157, 228, 179, 80, 52, 84, 57, 89, 359, 169, 404, 104,
                319, 275, 198, 200, 407, 310, 372, 289, 145, 71, 213, 99, 323, 132, 280, 424, 207,
                18, 201, 227, 68, 185, 66, 101, 63, 215, 364]) # random index
FREQ = 25
ID = 0


class model():
    def __init__(self, id=0,
                 # some recommended parameters
                 size=(256, 256),
                 size_reduce=(16, 16), # redundant size for registration
                 n_sigma=6,  # score of absolute normals factor0 = max(mean_std_lower_bound[0], E[x_normal] + Var[x_normal] * n_sigma), score(x_test) = score(x_test) - factor0
                 n_sigma_=10,  # scaling factor factor1 = max(mean_std_lower_bound[1], Var[x_normal] * n_sigma_), score(x_test) = (score(x_test) - factor0) / factor1
                 mean_std_lower_bound=(6, 5),  # minimum factor0 and factor1
                 top_roi_partition=0.005,
                 max_jitter=3,
                 # modified parameters below for your case
                 anchor_path='./regpadim/anchor/',
                 exp_path="./regpadim/IMGSAVE/anormal_det",
                 ckpt_pth="./regpadim/model_cov.pth",
                 ref_path="./regpadim/params_and_refs/ref_",
                 npz_path='./regpadim/params_and_refs/params_',
                 anchor_weight=0.6,  # anchor confidence, use for anchor-param & online-param fusion
                 n_hist=10, # video history frame for fast adaptation
                 lr=2e-2, # update learning rate
                 ROI = (0.1, 0.9, 0.1, 0.9), # numpy style
                 alarm_th=240,
                 alarm_len=2,
                 ):
        self.id = id
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Report the training process
        if not os.path.exists(exp_path + str(id)):
            os.makedirs(exp_path + str(id))

        self.size = size
        self.size_redundancy = size_reduce
        self.exp_path = exp_path + str(id) + "/"
        self.ref_path = ref_path + str(id) + ".jpg"
        self.npz_path = npz_path + str(id) + '.npz'
        self.anchor_path = anchor_path
        self.anchor_weight = anchor_weight
        self.n_hist = n_hist
        self.lr = lr
        self.n_sigma = n_sigma
        self.n_sigma_ = n_sigma_
        self.mean_std_lower_bound = mean_std_lower_bound
        self.alarm_th = alarm_th
        self.alarm_len = alarm_len
        self.image_size = (size[0] - 2 * self.size_redundancy[0], size[1] - 2 * self.size_redundancy[1])
        self.k_th = int(self.image_size[0] * self.image_size[1] // 64 * (1-top_roi_partition))
        self.lr_ = 1 - lr / n_hist if n_hist else 0.995
        self.rec = []
        self.q = []
        self.alarm_cnt0, self.counter = 0, 0
        self.base, self.tau = None, None
        self.mean, self.cov = None, None
        self.n_fea = len(IDX)
        self.filter_g, self.pad_g = gkern(5)
        self.filter_g = self.filter_g.to(self.dev)
        self.data_transform = get_data_transforms(size, ROI, max_jitter)

        # model
        self.model = unet([self.size_redundancy[1], self.size_redundancy[0], size[1], size[0]], 3)
        self.encoder = resnet18(pretrained=True)
        self.model.load_state_dict(torch.load(ckpt_pth))
        self.model.to(self.dev)
        self.encoder.to(self.dev)
        self.model.eval()
        self.encoder.eval()

        # dataset & transform & anchor
        if os.path.exists(self.ref_path) or os.path.exists(self.anchor_path + str(id)):
            if os.path.exists(self.ref_path):
                self.ref = cv2.imread(self.ref_path)
            else:
                self.ref = cv2.imread(glob.glob(self.anchor_path + str(id) + "/*")[0])
                cv2.imwrite(self.ref_path, self.ref)

            print("loading ref")
            self.ROI = (int(self.ref.shape[0]*ROI[0]), int(self.ref.shape[0]*ROI[1]), int(self.ref.shape[1]*ROI[2]), int(self.ref.shape[1]*ROI[3]))
            self.ref = cv2.resize(self.ref[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]], self.size)
            self.ref = (torch.from_numpy((cv2.cvtColor(self.ref, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(
                (2, 0, 1)) / 255 - MEAN) / STD).float()[None]).to(self.dev)
        else:
            self.ref, self.ROI = None, ROI

        self.init(id=id)

    def init(self, id):
        # set anchor
        if os.path.exists(self.npz_path): # init with previous parameter
            print("loading params_"+str(id)+".npz as parameters")
            params = np.load(self.npz_path)
            self.anchor_loaded = True
            self._mean, self._cov = params['mean'], params['cov'] # static variable
            self.mean, self.cov = self._mean.copy(), self._cov.copy() # on-line-updated variable
            self.base, self.tau = params['base'], params['tau'] # Scaling factor
        elif os.path.exists(self.anchor_path + str(id)): # init with anchor images
            anchor_dataset = AnchorLoader(self.anchor_path + str(id), self.data_transform)
            self.anchor_loaded = len(anchor_dataset)
            if self.anchor_loaded:
                anchor_batch = Data.DataLoader(anchor_dataset, batch_size=10, shuffle=False, num_workers=1, drop_last=False)
                print("estimating parameters with anchor")
                self._mean, self._cov, data = self.get_anchor_param(anchor_batch)
                self.mean, self.cov = self._mean.copy(), self._cov.copy()
                self.init_scaler(data)
                np.savez(self.npz_path, mean=self.mean, cov=self.cov, base=self.base, tau=self.tau)
        else:
            os.makedirs(self.anchor_path + str(id), exist_ok=True)
            self.anchor_loaded = False
        assert self.n_hist or self.anchor_loaded

        self.init_finished = self.anchor_loaded and not self.n_hist
        if self.init_finished:
            self.todevice()

    def init_scaler(self, data):
        score_map = self.compute_score(data)
        base, tau = np.mean(score_map) + self.n_sigma * np.std(score_map), np.std(score_map) * self.n_sigma_
        self.base, self.tau = max(base, self.mean_std_lower_bound[0]), max(tau, self.mean_std_lower_bound[1])

    def get_param(self):
        rec = np.asarray(self.rec)
        self.rec = []
        data = rec[:, :, :, IDX]
        B, H, W, C = data.shape
        data_flatten = data.reshape(B, H * W, C)
        mean = np.mean(data_flatten, axis=0)
        cov = np.zeros((H * W, C, C))
        I = np.identity(C)[None]
        for i in range(H * W):
            cov[i] = np.cov(data_flatten[:, i], rowvar=False) + 0.01 * I

        return mean, cov, data_flatten

    # do not use scipy.spatial.distance.mahalanobis,
    # or may cause undesired issue with Nvidia Jetson Platform and lower speed
    def compute_score(self, data, fuse=True):
        mean_fused, cov_fused = fuse_param(self._mean, self.mean, self._cov, self.cov, self.anchor_weight)
        delta = data - mean_fused
        score = delta.unsqueeze(1).bmm(torch.linalg.inv(cov_fused)).bmm(delta.unsqueeze(-1)).view((1, 1, self.image_size[1] // 8, self.image_size[0] // 8)) ** 0.5
        return nn.functional.conv2d(nn.functional.pad(score, (self.pad_g, self.pad_g, self.pad_g, self.pad_g), mode='reflect'), self.filter_g, groups=1)[0, 0]

    def get_anchor_param(self, anchor_loader):
        ref = self.ref[0][None]
        with torch.no_grad():
            for k, img in enumerate(anchor_loader):
                img = Variable(img).to(self.dev)
                fea = self.encoder(self.model(img, ref, test=True))
                self.rec.extend(fused_map(fea, (self.image_size[0] // 8, self.image_size[1] // 8)).cpu().numpy().astype(np.float16).transpose((0, 2, 3, 1)).tolist())

        return self.get_param()

    def get_fea(self):
        img = torch.from_numpy(np.concatenate(self.q, axis=0).transpose((0, 3, 1, 2)).astype(np.float32) / 255).float()
        self.q = []
        img = ((img - MEAN[None]) / STD[None]).float()

        with torch.no_grad():
            img = img.to(self.dev)
            img_ = self.model(img, self.ref, test=True)

            return fused_map(self.encoder(img_), (self.image_size[0] // 8, self.image_size[1] // 8))

    def todevice(self):
        self.mean, self.cov = torch.from_numpy(self.mean).float().to(self.dev), torch.from_numpy(self.cov).float().to(self.dev)
        self._mean, self._cov = torch.from_numpy(self._mean).float().to(self.dev), torch.from_numpy(self._cov).float().to(self.dev)

    def apply(self, frame, id):
        self.counter += 1
        results = []
        if self.ref is None:
            print("ref is not available, use and save the first frame as ref"%id)
            self.ref = frame.copy()
            cv2.imwrite(self.ref_path, self.ref)
            self.ROI = (int(self.ref.shape[0] * self.ROI[0]), int(self.ref.shape[0] * self.ROI[1]), int(self.ref.shape[1] * self.ROI[2]), int(self.ref.shape[1] * self.ROI[3]))
            self.ref = cv2.resize(self.ref[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]], self.size)
            self.ref = (torch.from_numpy((cv2.cvtColor(self.ref, cv2.COLOR_BGR2RGB).astype(np.float32).transpose((2, 0, 1)) / 255 - MEAN) / STD).float()[None]).to(self.dev)

        if self.init_finished:
            # read data
            frame_ = cv2.resize(frame[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]], self.size)
            self.q.append(cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)[None])

            # infer
            data_flatten = self.get_fea()[:, IDX].permute((0, 2, 3, 1)).view((self.image_size[0] * self.image_size[1] // 64, -1))
            self.mean, self.cov = fuse_param(self.mean, data_flatten, self.cov, w1=self.lr_) # update on-line paramter
            score_map = self.compute_score(data_flatten).cpu().numpy()

            # rescale final anomaly score
            scores = np.minimum(1, np.maximum(0, score_map - self.base) / self.tau)* 255
            scores = scores.astype(np.uint8)
            score = np.partition(scores.flatten(), self.k_th)[self.k_th:].mean()
            res = np.where((scores < 180) * (scores > 40), 1, 0).sum() / (np.where(scores > 180, 1, 0).sum() + 1)
            all_idx_h, all_idx_w = np.where(scores>self.alarm_th)
            score_all = scores[all_idx_h.min():all_idx_h.max()+1, all_idx_w.min():all_idx_w.max()+1].mean() if len(all_idx_h) else 0

            # SET_YOUR_OWN_DETECTION_THRESHOLD
            if score > self.alarm_th and scores.mean() > 5 and ((res < 1.5 and scores.mean() < 35) or (res < 1 and score_all>self.alarm_th*0.8)):
                curr_time = datetime.datetime.now()
                savepath = os.path.join(self.exp_path, str(curr_time.minute) + "_" + str(curr_time.second) + "_" + str(curr_time.microsecond))

                self.alarm_cnt0 += 1
                if self.alarm_cnt0 > self.alarm_len:
                    anopos = np.where(scores > scores.max()*0.8)
                    H, W = self.ROI[1]-self.ROI[0], self.ROI[3]-self.ROI[2]
                    luh = (H * (anopos[0].min() * 8 + self.size_redundancy[1]) / self.size[1] + self.ROI[0]).astype(np.int64)
                    luw = (W * (anopos[1].min() * 8 + self.size_redundancy[0]) / self.size[0] + self.ROI[2]).astype(np.int64)
                    rdh = (H * ((anopos[0].max()+1) * 8 + self.size_redundancy[1]) / self.size[1] + self.ROI[0]).astype(np.int64)
                    rdw = (W * ((anopos[1].max()+1) * 8 + self.size_redundancy[0]) / self.size[0] + self.ROI[2]).astype(np.int64)
                    cv2.imwrite(savepath + str(self.counter) + '_alarm.jpg', cv2.rectangle(frame, (luw, luh), (rdw, rdh), 255, 10))
                    results.append(['abnormal', luh, luw, rdh - luh, rdw - luw])
                else:
                    results.append(['normal', 0, 0, 0, 0])
            else:
                self.alarm_cnt0 = max(0, self.alarm_cnt0 - 1)
                results.append(['normal', 0, 0, 0, 0])
            return results

        else:
            if self.anchor_loaded:
                frame_ = cv2.resize(frame[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]], self.size)
                self.q.append(cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)[None])
            else:
                cv2.imwrite(self.anchor_path + str(id) + "/" + str(self.counter) + ".png", frame)

            if self.counter > self.n_hist:
                if self.anchor_loaded:
                    self.rec.extend(self.get_fea().cpu().numpy().astype(np.float16).transpose((0, 2, 3, 1)).tolist())
                    self.mean, self.cov, update_data = self.get_param()
                else:
                    print("estimating parameters and save to params_%i.npz" % id)
                    params_path = self.npz_path + "%i.npz" % id
                    self.anchor_loaded = True
                    anchor_dataset = AnchorLoader(self.anchor_path + str(id), self.data_transform)
                    anchor_batch = Data.DataLoader(anchor_dataset, batch_size=10, shuffle=False, num_workers=1, drop_last=False)
                    self._mean, self._cov, data = self.get_anchor_param(anchor_batch)
                    self.mean, self.cov = self._mean.copy(), self._cov.copy()
                    self.init_scaler(data)
                    np.savez(params_path, mean=self.mean, cov=self.cov, base=self.base, tau=self.tau)
                    self.todevice()
                    self.init_finished = True
                    return

                self.init_scaler(update_data)
                self.todevice()
                self.init_finished = True


if __name__=='__main__':
    detector = model(ID)
    while True:
        cap = cv2.VideoCapture(ID)
        if cap.isOpened():
            print('VideoCapture get')
        else:
            print('VideoCapture fail')
            cap.release()
            continue

    i = 0
    while True:
        flag, frame = cap.read()
        if flag and frame is not None:
            if i%FREQ == 0:
                print(detector.apply(frame, ID))
            i += 1
