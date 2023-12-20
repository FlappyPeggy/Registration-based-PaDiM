from anormal_det_video import *

class model_nohist(model):
    def __init__(self,id=-1,
                 # some recommended parameters
                 size=(256, 256),
                 size_reduce=(16,16), # redundant size for registration
                 n_sigma=6, # score of absolute normals factor0 = max(mean_std_lower_bound[0], E[x_normal] + Var[x_normal] * n_sigma), score(x_test) = score(x_test) - factor0
                 n_sigma_=10, # scaling factor factor1 = max(mean_std_lower_bound[1], Var[x_normal] * n_sigma_), score(x_test) = (score(x_test) - factor0) / factor1
                 mean_std_lower_bound=(6, 5),  # minimum factor0 and factor1
                 top_roi_partition=4e-5,
                 max_jitter=1,
                 # modified parameters below for your case
                 ROI = (0.1, 0.9, 0.1, 0.9), # numpy style
                 alarm_th=240,
                 ):
        super().__init__(id=id, size=size, size_reduce=size_reduce, n_hist=0, alarm_len=0, top_roi_partition=top_roi_partition,
                 anchor_weight=1., n_sigma=n_sigma, n_sigma_=n_sigma_, mean_std_lower_bound=mean_std_lower_bound,
                 max_jitter=max_jitter, ROI = ROI, alarm_th=alarm_th,
                 )
        assert self.ref is not None, "you have to set ref image before inference"

    def apply(self, frame, *args):
        if isinstance(frame,str): frame = cv2.imread(frame, 0)
        self.counter += 1

        frame_ = cv2.resize(frame[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]], self.size)
        self.q.append(cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)[None])

        score_map = self.compute_score(self.get_fea()[:, IDX].permute((0, 2, 3, 1)).view((self.N, -1))).cpu().numpy()

        scores = np.minimum(1, np.maximum(0, score_map - self.base) / self.tau)* 255
        scores = scores.astype(np.uint8)
        score = np.partition(scores.flatten(), self.k_th)[self.k_th:].mean()
        res = np.where((scores < self.alarm_th*0.85) * (scores > self.alarm_th*0.6), 1, 0).sum() / (np.where(scores > self.alarm_th*0.9, 1, 0).sum() + 1e-4)
        all_idx_h, all_idx_w = np.where(scores>scores.max()*0.8)
        sub_scores = scores[all_idx_h.min():all_idx_h.max()+1, all_idx_w.min():all_idx_w.max()+1] if len(all_idx_h)>4 else np.ones_like(scores)*self.alarm_th*0.7
        res_ = np.where((sub_scores < self.alarm_th*0.8) * (sub_scores > self.alarm_th*0.5), 1, 0).sum() / (np.where(sub_scores > self.alarm_th*0.9, 1, 0).sum() + 1)
        area = np.where(scores > self.alarm_th*0.85, 255, 0).mean()

        # SET_YOUR_OWN_DETECTION_THRESHOLD
        if score > self.alarm_th and ((area > 5 and res<1.65) or (area > 2.5 and res < 3.1 and res_ < 0.45)):
            curr_time = datetime.datetime.now()
            savepath = os.path.join(self.exp_path, str(curr_time.minute) + "_" + str(curr_time.second) + "_" + str(curr_time.microsecond))
            anopos = np.where(scores > scores.max()*0.8)
            H, W = self.ROI[1]-self.ROI[0], self.ROI[3]-self.ROI[2]
            luh = (H * (anopos[0].min() * 8 + self.size_redundancy[1]) / self.size[1] + self.ROI[0]).astype(np.int64)
            luw = (W * (anopos[1].min() * 8 + self.size_redundancy[0]) / self.size[0] + self.ROI[2]).astype(np.int64)
            rdh = (H * ((anopos[0].max() + 1) * 8 + self.size_redundancy[1]) / self.size[1] + self.ROI[0]).astype(np.int64)
            rdw = (W * ((anopos[1].max() + 1) * 8 + self.size_redundancy[0]) / self.size[0] + self.ROI[2]).astype(np.int64)
            cv2.imwrite(savepath + str(self.counter) + '_alarm.jpg', cv2.rectangle(frame, (luw, luh), (rdw,rdh), 255, 10))
            return ['abnormal', luh, luw, rdh - luh, rdw - luw]
        else:
            return ['normal', 0, 0, 0, 0]

if __name__=='__main__':
    detector = model_nohist(ID)
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
            if YOUR_OWN_TRIGGER_FLAG:
                print(detector.apply(frame, ID))
            i += 1
